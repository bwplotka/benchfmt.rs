// Copyright 2026 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Global allocator wrapper for reporting `B/op` and `allocs/op` in benchmarks.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! {
    // `const` initialization guarantees TLS access never triggers a recursive heap allocation.
    static THREAD_ALLOC_COUNTS: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
    static THREAD_ALLOC_PAUSED: Cell<bool> = const { Cell::new(false) };
}

#[inline]
fn is_alloc_tracking_paused() -> bool {
    THREAD_ALLOC_PAUSED.try_with(Cell::get).unwrap_or(false)
}

/// Runs `f` while pausing [`CountingAlloc`], [`ProcessAlloc`], and `-memprofile` sampling on the
/// calling thread so internal harness bookkeeping (such as [`crate::Bencher::report_metric`])
/// never skews `B/op` or `allocs/op`.
pub(crate) fn without_alloc_tracking<R>(f: impl FnOnce() -> R) -> R {
    struct PauseGuard(bool);
    impl Drop for PauseGuard {
        fn drop(&mut self) {
            let _ = THREAD_ALLOC_PAUSED.try_with(|c| c.set(self.0));
        }
    }
    let prev = THREAD_ALLOC_PAUSED
        .try_with(|c| {
            let old = c.get();
            c.set(true);
            old
        })
        .unwrap_or(false);
    let _guard = PauseGuard(prev);
    f()
}

#[inline]
fn record_thread_alloc(bytes: usize) {
    if is_alloc_tracking_paused() {
        return;
    }
    #[cfg(feature = "pprof")]
    if crate::pprof_support::is_in_heap_sample() {
        return;
    }
    let _ = THREAD_ALLOC_COUNTS.try_with(|cell| {
        let (a, b) = cell.get();
        cell.set((a.wrapping_add(1), b.wrapping_add(bytes as u64)));
    });
    #[cfg(feature = "pprof")]
    crate::pprof_support::maybe_record_heap_alloc(bytes);
}

/// Point-in-time snapshot of cumulative heap allocation counters from an [`AllocTracker`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocSnapshot {
    /// Total number of heap allocation calls observed so far by the tracker.
    pub allocs: u64,
    /// Total number of heap bytes requested so far by the tracker.
    pub bytes: u64,
}

impl AllocSnapshot {
    /// Computes the difference between `after` and `self` using wrapping subtraction.
    #[must_use]
    pub fn delta_to(self, after: Self) -> Self {
        Self {
            allocs: after.allocs.wrapping_sub(self.allocs),
            bytes: after.bytes.wrapping_sub(self.bytes),
        }
    }
}

/// Trait implemented by allocation trackers attached to a [`crate::Suite`].
pub trait AllocTracker: Sync {
    /// Captures the tracker's current cumulative allocation counters (thread-local for
    /// [`CountingAlloc`], process-wide for [`ProcessAlloc`]).
    fn snapshot(&self) -> AllocSnapshot;
}

/// Thin [`GlobalAlloc`] wrapper that counts heap allocations and allocated bytes per thread
/// without atomic cache-line contention or cross-thread noise, matching Go's
/// `testing.B.ReportAllocs` (`B/op` and `allocs/op`) for single-threaded benchmarks.
pub struct CountingAlloc<A = System> {
    inner: A,
}

impl CountingAlloc<System> {
    /// Creates a [`CountingAlloc`] wrapping [`std::alloc::System`].
    #[must_use]
    pub const fn new() -> Self {
        Self::with_allocator(System)
    }
}

impl Default for CountingAlloc<System> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> CountingAlloc<A> {
    /// Creates a [`CountingAlloc`] wrapping a custom [`GlobalAlloc`] implementation.
    #[must_use]
    pub const fn with_allocator(inner: A) -> Self {
        Self { inner }
    }

    /// Captures the current cumulative allocation counters on the calling thread.
    #[must_use]
    pub fn snapshot(&self) -> AllocSnapshot {
        THREAD_ALLOC_COUNTS
            .try_with(|cell| {
                let (allocs, bytes) = cell.get();
                AllocSnapshot { allocs, bytes }
            })
            .unwrap_or_default()
    }
}

impl<A: Sync> AllocTracker for CountingAlloc<A> {
    fn snapshot(&self) -> AllocSnapshot {
        Self::snapshot(self)
    }
}

// SAFETY: Every allocation call delegates directly to `self.inner` with identical layout and
// pointer arguments, only updating `const`-initialized thread-local counters on success.
unsafe impl<A: GlobalAlloc> GlobalAlloc for CountingAlloc<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Upheld by the caller of `GlobalAlloc::alloc`.
        let ptr = unsafe { self.inner.alloc(layout) };
        if !ptr.is_null() {
            record_thread_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Upheld by the caller of `GlobalAlloc::dealloc`.
        unsafe { self.inner.dealloc(ptr, layout) };
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Upheld by the caller of `GlobalAlloc::alloc_zeroed`.
        // Overridden explicitly so inner allocators that delegate `alloc_zeroed` to `alloc`
        // do not double-count.
        let ptr = unsafe { self.inner.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record_thread_alloc(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old_addr = ptr.addr();
        // SAFETY: Upheld by the caller of `GlobalAlloc::realloc`.
        let new_ptr = unsafe { self.inner.realloc(ptr, layout, new_size) };
        // NOTE: Always record growth deterministically to match Go's slice append behavior, while
        // ignoring in-place shrinks (`shrink_to_fit` / `into_boxed_slice`) that reuse `ptr`.
        if !new_ptr.is_null() && (new_size > layout.size() || new_ptr.addr() != old_addr) {
            record_thread_alloc(new_size);
        }
        new_ptr
    }
}

/// Process-wide [`GlobalAlloc`] wrapper that counts heap allocations and allocated bytes across
/// all threads using relaxed atomics.
///
/// Prefer [`CountingAlloc`] for single-threaded benchmarks to avoid atomic cache-line traffic and
/// background-thread noise. Use [`ProcessAlloc`] when a benchmark dispatches work onto worker
/// threads, such as a multi-threaded Tokio runtime or Rayon pool, during sequential
/// [`crate::Suite`] epochs so worker allocations are included in `B/op` and `allocs/op`.
pub struct ProcessAlloc<A = System> {
    inner: A,
    allocs: AtomicU64,
    bytes: AtomicU64,
}

impl ProcessAlloc<System> {
    /// Creates a [`ProcessAlloc`] wrapping [`std::alloc::System`].
    #[must_use]
    pub const fn new() -> Self {
        Self::with_allocator(System)
    }
}

impl Default for ProcessAlloc<System> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> ProcessAlloc<A> {
    /// Creates a [`ProcessAlloc`] wrapping a custom [`GlobalAlloc`] implementation.
    #[must_use]
    pub const fn with_allocator(inner: A) -> Self {
        Self {
            inner,
            allocs: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }

    /// Captures the current cumulative process-wide allocation counters.
    #[must_use]
    pub fn snapshot(&self) -> AllocSnapshot {
        AllocSnapshot {
            allocs: self.allocs.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
        }
    }

    #[inline]
    fn record_alloc(&self, bytes: usize) {
        if is_alloc_tracking_paused() {
            return;
        }
        #[cfg(feature = "pprof")]
        if crate::pprof_support::is_in_heap_sample() {
            return;
        }
        self.allocs.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        #[cfg(feature = "pprof")]
        crate::pprof_support::maybe_record_heap_alloc(bytes);
    }
}

impl<A: Sync> AllocTracker for ProcessAlloc<A> {
    fn snapshot(&self) -> AllocSnapshot {
        Self::snapshot(self)
    }
}

// SAFETY: Every allocation call delegates directly to `self.inner` with identical layout and
// pointer arguments, only updating relaxed atomic counters on success.
unsafe impl<A: GlobalAlloc> GlobalAlloc for ProcessAlloc<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Upheld by the caller of `GlobalAlloc::alloc`.
        let ptr = unsafe { self.inner.alloc(layout) };
        if !ptr.is_null() {
            self.record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Upheld by the caller of `GlobalAlloc::dealloc`.
        unsafe { self.inner.dealloc(ptr, layout) };
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Upheld by the caller of `GlobalAlloc::alloc_zeroed`.
        // Overridden explicitly so inner allocators that delegate `alloc_zeroed` to `alloc`
        // do not double-count.
        let ptr = unsafe { self.inner.alloc_zeroed(layout) };
        if !ptr.is_null() {
            self.record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old_addr = ptr.addr();
        // SAFETY: Upheld by the caller of `GlobalAlloc::realloc`.
        let new_ptr = unsafe { self.inner.realloc(ptr, layout, new_size) };
        // NOTE: Always record growth deterministically to match Go's slice append behavior, while
        // ignoring in-place shrinks (`shrink_to_fit` / `into_boxed_slice`) that reuse `ptr`.
        if !new_ptr.is_null() && (new_size > layout.size() || new_ptr.addr() != old_addr) {
            self.record_alloc(new_size);
        }
        new_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubRealloc {
        relocate_on_shrink: bool,
        fail: bool,
    }

    unsafe impl GlobalAlloc for StubRealloc {
        unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
            if self.fail {
                std::ptr::null_mut()
            } else {
                std::ptr::without_provenance_mut(0x1000)
            }
        }
        unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            unsafe { self.alloc(layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            if self.fail {
                return std::ptr::null_mut();
            }
            if new_size < layout.size() && self.relocate_on_shrink {
                std::ptr::without_provenance_mut(ptr.addr() + 0x1000)
            } else {
                ptr
            }
        }
    }

    fn assert_tracker_realloc_and_zeroed<T: GlobalAlloc + AllocTracker>(
        make: impl Fn(StubRealloc) -> T,
    ) {
        let in_place = make(StubRealloc {
            relocate_on_shrink: false,
            fail: false,
        });
        let layout = Layout::from_size_align(128, 8).unwrap();
        let s0 = AllocTracker::snapshot(&in_place);
        let ptr = unsafe { in_place.alloc(layout) };
        let _ = unsafe { in_place.alloc_zeroed(layout) };
        // In-place growth (`new_size > layout.size()`) must be counted even when `new_ptr == ptr`.
        let ptr = unsafe { in_place.realloc(ptr, layout, 256) };
        let grown_layout = Layout::from_size_align(256, 8).unwrap();
        // Same-size and in-place shrink (`new_size <= layout.size()` and `new_ptr == ptr`) must be
        // ignored.
        let ptr = unsafe { in_place.realloc(ptr, grown_layout, 256) };
        let _ = unsafe { in_place.realloc(ptr, grown_layout, 64) };
        assert_eq!(
            s0.delta_to(AllocTracker::snapshot(&in_place)),
            AllocSnapshot {
                allocs: 3,
                bytes: 128 + 128 + 256,
            }
        );

        let relocating = make(StubRealloc {
            relocate_on_shrink: true,
            fail: false,
        });
        let s1 = AllocTracker::snapshot(&relocating);
        let _ = unsafe { relocating.realloc(ptr, grown_layout, 64) };
        assert_eq!(
            s1.delta_to(AllocTracker::snapshot(&relocating)),
            AllocSnapshot {
                allocs: 1,
                bytes: 64,
            }
        );

        let failing = make(StubRealloc {
            relocate_on_shrink: false,
            fail: true,
        });
        let s2 = AllocTracker::snapshot(&failing);
        assert!(unsafe { failing.alloc(layout) }.is_null());
        assert!(unsafe { failing.alloc_zeroed(layout) }.is_null());
        assert!(unsafe { failing.realloc(ptr, grown_layout, 512) }.is_null());
        assert_eq!(
            s2.delta_to(AllocTracker::snapshot(&failing)),
            AllocSnapshot::default()
        );
    }

    #[test]
    fn realloc_counts_growth_and_relocating_shrink_but_ignores_in_place_shrink() {
        assert_tracker_realloc_and_zeroed(CountingAlloc::with_allocator);
        assert_tracker_realloc_and_zeroed(ProcessAlloc::with_allocator);
    }
}
