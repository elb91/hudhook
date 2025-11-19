//! Hooks for DirectX 12.
//! 
//! based on proven stable D3D12Hook.cpp implementation
//! Changes:
//! 1. Check queue type is DIRECT in ExecuteCommandLists
//! 2. Reset pipeline on ResizeBuffers (handles scene transitions)
//! 3. Proper initialization context reset
//! 4. Manual vtable offset verification

use std::ffi::c_void;
use std::mem;
use std::sync::atomic::Ordering;
use std::sync::OnceLock;

use imgui::Context;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use tracing::{debug, error, trace, warn};
use windows::core::{Error, Interface, Result, HRESULT};
use windows::Win32::Foundation::BOOL;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12CreateDevice, ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Resource,
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_MODE_DESC, DXGI_MODE_SCALING_UNSPECIFIED,
    DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED, DXGI_RATIONAL, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, IDXGIFactory2, IDXGISwapChain, IDXGISwapChain3, DXGI_SWAP_CHAIN_DESC,
    DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH, DXGI_SWAP_EFFECT_FLIP_DISCARD,
    DXGI_USAGE_RENDER_TARGET_OUTPUT,
};

use super::DummyHwnd;
use crate::mh::MhHook;
use crate::renderer::{D3D12RenderEngine, Pipeline};
use crate::{perform_eject, util, Hooks, ImguiRenderLoop, EJECT_REQUESTED, HOOK_EJECTION_BARRIER};

type DXGISwapChainPresentType =
    unsafe extern "system" fn(this: IDXGISwapChain3, sync_interval: u32, flags: u32) -> HRESULT;

type DXGISwapChainResizeBuffersType = unsafe extern "system" fn(
    this: IDXGISwapChain3,
    buffer_count: u32,
    width: u32,
    height: u32,
    new_format: DXGI_FORMAT,
    flags: u32,
) -> HRESULT;

type D3D12CommandQueueExecuteCommandListsType = unsafe extern "system" fn(
    this: ID3D12CommandQueue,
    num_command_lists: u32,
    command_lists: *mut ID3D12CommandList,
);

struct Trampolines {
    dxgi_swap_chain_present: DXGISwapChainPresentType,
    dxgi_swap_chain_resize_buffers: DXGISwapChainResizeBuffersType,
    d3d12_command_queue_execute_command_lists: D3D12CommandQueueExecuteCommandListsType,
}

static mut TRAMPOLINES: OnceLock<Trampolines> = OnceLock::new();

enum InitializationContext {
    Empty,
    WithSwapChain(IDXGISwapChain3),
    Complete(IDXGISwapChain3, ID3D12CommandQueue),
    Done,
}

impl InitializationContext {
    // Transition to a state where the swap chain is set. Ignore other mutations.
    fn insert_swap_chain(&mut self, swap_chain: &IDXGISwapChain3) {
        *self = match mem::replace(self, InitializationContext::Empty) {
            InitializationContext::Empty => {
                InitializationContext::WithSwapChain(swap_chain.clone())
            },
            s => s,
        }
    }

    // Check that command queue is DIRECT type before accepting it
  fn insert_command_queue(&mut self, command_queue: &ID3D12CommandQueue) {
    *self = match mem::replace(self, InitializationContext::Empty) {
        InitializationContext::WithSwapChain(swap_chain) => {
            let desc = unsafe { command_queue.GetDesc() };
            
            if desc.Type != D3D12_COMMAND_LIST_TYPE_DIRECT {
                debug!("Ignoring non-DIRECT command queue");
                *self = InitializationContext::WithSwapChain(swap_chain);
                return;
            }
            
            //Just accept the first DIRECT queue (like D3D12Hook.cpp)
            // Don't try to validate it's in the swap chain struct
            debug!("Captured DIRECT command queue: {:?}", command_queue);
            InitializationContext::Complete(swap_chain, command_queue.clone())
        },
        s => s,
    }
}

    // Retrieve the values if the context is complete.
    fn get(&self) -> Option<(IDXGISwapChain3, ID3D12CommandQueue)> {
        if let InitializationContext::Complete(swap_chain, command_queue) = self {
            Some((swap_chain.clone(), command_queue.clone()))
        } else {
            None
        }
    }

    // Mark the context as done so no further operations are executed on it.
    fn done(&mut self) {
        if let InitializationContext::Complete(..) = self {
            *self = InitializationContext::Done;
        }
    }
    
    // Add reset method 
    fn reset(&mut self) {
        *self = InitializationContext::Empty;
    }

    unsafe fn check_command_queue(
        swap_chain: &IDXGISwapChain3,
        command_queue: &ID3D12CommandQueue,
    ) -> bool {
        let swap_chain_ptr = swap_chain.as_raw() as *mut *mut c_void;
        let readable_ptrs = util::readable_region(swap_chain_ptr, 512);

        match readable_ptrs.iter().position(|&ptr| std::ptr::eq(ptr, command_queue.as_raw())) {
            Some(idx) => {
                debug!(
                    "Found command queue pointer in swap chain struct at offset +0x{:x}",
                    idx * mem::size_of::<usize>(),
                );
                true
            },
            None => {
                warn!(
                    "Couldn't find command queue pointer in swap chain struct ({} out of 512 \
                     pointers were readable)",
                    readable_ptrs.len()
                );
                false
            },
        }
    }
}

static INITIALIZATION_CONTEXT: Mutex<InitializationContext> =
    Mutex::new(InitializationContext::Empty);
static mut PIPELINE: OnceCell<Mutex<Pipeline<D3D12RenderEngine>>> = OnceCell::new();
static mut RENDER_LOOP: OnceCell<Box<dyn ImguiRenderLoop + Send + Sync>> = OnceCell::new();

unsafe fn init_pipeline() -> Result<Mutex<Pipeline<D3D12RenderEngine>>> {
    let Some((swap_chain, command_queue)) = ({ INITIALIZATION_CONTEXT.lock().get() }) else {
        error!("Initialization context incomplete");
        return Err(Error::from_hresult(HRESULT(-1)));
    };

    let hwnd = util::try_out_param(|v| swap_chain.GetDesc(v)).map(|desc| desc.OutputWindow)?;

    let mut ctx = Context::create();
    let engine = D3D12RenderEngine::new(&command_queue, &mut ctx)?;

    let Some(render_loop) = RENDER_LOOP.take() else {
        error!("Render loop not yet initialized");
        return Err(Error::from_hresult(HRESULT(-1)));
    };

    let pipeline = Pipeline::new(hwnd, ctx, engine, render_loop).map_err(|(e, render_loop)| {
        RENDER_LOOP.get_or_init(move || render_loop);
        e
    })?;

    {
        INITIALIZATION_CONTEXT.lock().done();
    }

    Ok(Mutex::new(pipeline))
}

fn render(swap_chain: &IDXGISwapChain3) -> Result<()> {
    unsafe {
        let pipeline = PIPELINE.get_or_try_init(|| init_pipeline())?;

        let Some(mut pipeline) = pipeline.try_lock() else {
            error!("Could not lock pipeline");
            return Err(Error::from_hresult(HRESULT(-1)));
        };

        pipeline.prepare_render()?;

        let target: ID3D12Resource =
            swap_chain.GetBuffer(swap_chain.GetCurrentBackBufferIndex())?;

        pipeline.render(target)?;
    }

    Ok(())
}

unsafe extern "system" fn dxgi_swap_chain_present_impl(
    swap_chain: IDXGISwapChain3,
    sync_interval: u32,
    flags: u32,
) -> HRESULT {
    let _hook_ejection_guard = HOOK_EJECTION_BARRIER.acquire_ejection_guard();
    
    {
        INITIALIZATION_CONTEXT.lock().insert_swap_chain(&swap_chain);
    }

    let Trampolines { dxgi_swap_chain_present, .. } =
        TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");

    // Check if initialization is complete
    let is_ready = {
        let ctx = INITIALIZATION_CONTEXT.lock();
        matches!(&*ctx, InitializationContext::Complete(_, _) | InitializationContext::Done)
    };
    
    if is_ready {
        if let Err(e) = render(&swap_chain) {
            util::print_dxgi_debug_messages();
            error!("Render error: {e:?}");
        }
    }

    trace!("Call IDXGISwapChain::Present trampoline");
    let result = dxgi_swap_chain_present(swap_chain, sync_interval, flags);

    if EJECT_REQUESTED.load(Ordering::SeqCst) {
        perform_eject();
    }

    result
}

// Complete reset on ResizeBuffers (handles scene transitions)
unsafe extern "system" fn dxgi_swap_chain_resize_buffers_impl(
    p_this: IDXGISwapChain3,
    buffer_count: u32,
    width: u32,
    height: u32,
    new_format: DXGI_FORMAT,
    flags: u32,
) -> HRESULT {
    let _hook_ejection_guard = HOOK_EJECTION_BARRIER.acquire_ejection_guard();
    
    error!(
        "!!! ResizeBuffers called !!! buffer_count={}, width={}, height={}, format={:?}",
        buffer_count, width, height, new_format
    );
    
    debug!("ResizeBuffers - resetting pipeline");
    
    // Reset everything
    if let Some(pipeline) = PIPELINE.take() {
        debug!("Dropping existing pipeline");
        drop(pipeline);
    }
    
    {
        let mut ctx = INITIALIZATION_CONTEXT.lock();
        ctx.reset();
        debug!("Initialization context reset");
    }
    
    let Trampolines { dxgi_swap_chain_resize_buffers, .. } =
        TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");

    debug!("Calling original ResizeBuffers");
    let result = dxgi_swap_chain_resize_buffers(p_this, buffer_count, width, height, new_format, flags);
    debug!("ResizeBuffers returned: {:?}", result);
    
    result
}

unsafe extern "system" fn d3d12_command_queue_execute_command_lists_impl(
    command_queue: ID3D12CommandQueue,
    num_command_lists: u32,
    command_lists: *mut ID3D12CommandList,
) {
    let _hook_ejection_guard = HOOK_EJECTION_BARRIER.acquire_ejection_guard();
    trace!(
        "ID3D12CommandQueue::ExecuteCommandLists({command_queue:?}, {num_command_lists}, \
         {command_lists:p}) invoked",
    );

    {
        // is applied in insert_command_queue - it checks queue type
        INITIALIZATION_CONTEXT.lock().insert_command_queue(&command_queue);
    }

    let Trampolines { d3d12_command_queue_execute_command_lists, .. } =
        TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");

    d3d12_command_queue_execute_command_lists(command_queue, num_command_lists, command_lists);
}

fn get_target_addrs() -> (
    DXGISwapChainPresentType,
    DXGISwapChainResizeBuffersType,
    D3D12CommandQueueExecuteCommandListsType,
) {
    let dummy_hwnd = DummyHwnd::new();

    let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(0) }.unwrap();
    let adapter = unsafe { factory.EnumAdapters(0) }.unwrap();

    let device: ID3D12Device =
        util::try_out_ptr(|v| unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, v) })
            .expect("D3D12CreateDevice failed");

    let command_queue: ID3D12CommandQueue = unsafe {
        device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Priority: 0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        })
    }
    .unwrap();

    let swap_chain: IDXGISwapChain = match util::try_out_ptr(|v| unsafe {
        factory
            .CreateSwapChain(
                &command_queue,
                &DXGI_SWAP_CHAIN_DESC {
                    BufferDesc: DXGI_MODE_DESC {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        ScanlineOrdering: DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED,
                        Scaling: DXGI_MODE_SCALING_UNSPECIFIED,
                        Width: 640,
                        Height: 480,
                        RefreshRate: DXGI_RATIONAL { Numerator: 60, Denominator: 1 },
                    },
                    BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                    BufferCount: 2,
                    OutputWindow: dummy_hwnd.hwnd(),
                    Windowed: BOOL(1),
                    SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Flags: DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH.0 as _,
                },
                v,
            )
            .ok()
    }) {
        Ok(swap_chain) => swap_chain,
        Err(e) => {
            util::print_dxgi_debug_messages();
            panic!("{e:?}");
        },
    };

    // Manual vtable extraction with verified offsets
    // Based on D3D12Hook.cpp
    unsafe {
        // Get raw vtable pointers
        let sc_vtable = *(swap_chain.as_raw() as *const *const usize);
        let q_vtable = *(command_queue.as_raw() as *const *const usize);
        
        //  Use correct, verified offsets
        // IDXGISwapChain vtable offsets:
        //   0-2: IUnknown (QueryInterface, AddRef, Release)
        //   3-7: IDXGIObject methods
        //   8: Present         ← Offset 8
        //   9: GetBuffer
        //   10-12: Other methods
        //   13: ResizeBuffers  ← Offset 13
        //
        // ID3D12CommandQueue vtable offsets:
        //   0-2: IUnknown
        //   3-9: ID3D12Object + ID3D12DeviceChild
        //   10: ExecuteCommandLists  ← Offset 10
        
        let present_ptr = *sc_vtable.add(8);
        let resize_buffers_ptr = *sc_vtable.add(13);
        let execute_command_lists_ptr = *q_vtable.add(10);
        
        debug!("Vtable addresses extracted:");
        debug!("  Present:              {:p}", present_ptr as *const c_void);
        debug!("  ResizeBuffers:        {:p}", resize_buffers_ptr as *const c_void);
        debug!("  ExecuteCommandLists:  {:p}", execute_command_lists_ptr as *const c_void);
        
        // Verify these match what windows-rs would give us
        let windows_present = swap_chain.vtable().Present as *const c_void;
        let windows_resize = swap_chain.vtable().ResizeBuffers as *const c_void;
        let windows_execute = command_queue.vtable().ExecuteCommandLists as *const c_void;
        
        if present_ptr as *const c_void != windows_present {
            warn!(
                "Present vtable mismatch! Manual: {:p}, Windows-rs: {:p}",
                present_ptr as *const c_void, windows_present
            );
        }
        if resize_buffers_ptr as *const c_void != windows_resize {
            warn!(
                "ResizeBuffers vtable mismatch! Manual: {:p}, Windows-rs: {:p}",
                resize_buffers_ptr as *const c_void, windows_resize
            );
        }
        if execute_command_lists_ptr as *const c_void != windows_execute {
            warn!(
                "ExecuteCommandLists vtable mismatch! Manual: {:p}, Windows-rs: {:p}",
                execute_command_lists_ptr as *const c_void, windows_execute
            );
        }
        
        (
            mem::transmute(present_ptr),
            mem::transmute(resize_buffers_ptr),
            mem::transmute(execute_command_lists_ptr),
        )
    }
}

/// Hooks for DirectX 12.
pub struct ImguiDx12Hooks([MhHook; 3]);

impl ImguiDx12Hooks {
    /// Construct a set of [`MhHook`]s that will render UI via the
    /// provided [`ImguiRenderLoop`].
    ///
    /// The following functions are hooked:
    /// - `IDXGISwapChain3::Present`
    /// - `IDXGISwapChain3::ResizeBuffers`
    /// - `ID3D12CommandQueue::ExecuteCommandLists`
    ///
    /// # Safety
    ///
    /// yolo
    pub unsafe fn new<T>(t: T) -> Self
    where
        T: ImguiRenderLoop + Send + Sync + 'static,
    {
        let (
            dxgi_swap_chain_present_addr,
            dxgi_swap_chain_resize_buffers_addr,
            d3d12_command_queue_execute_command_lists_addr,
        ) = get_target_addrs();

        trace!("IDXGISwapChain::Present = {:p}", dxgi_swap_chain_present_addr as *const c_void);
        let hook_present = MhHook::new(
            dxgi_swap_chain_present_addr as *mut _,
            dxgi_swap_chain_present_impl as *mut _,
        )
        .expect("couldn't create IDXGISwapChain::Present hook");
        let hook_resize_buffers = MhHook::new(
            dxgi_swap_chain_resize_buffers_addr as *mut _,
            dxgi_swap_chain_resize_buffers_impl as *mut _,
        )
        .expect("couldn't create IDXGISwapChain::ResizeBuffers hook");
        let hook_cqecl = MhHook::new(
            d3d12_command_queue_execute_command_lists_addr as *mut _,
            d3d12_command_queue_execute_command_lists_impl as *mut _,
        )
        .expect("couldn't create ID3D12CommandQueue::ExecuteCommandLists hook");

        RENDER_LOOP.get_or_init(|| Box::new(t));

        TRAMPOLINES.get_or_init(|| Trampolines {
            dxgi_swap_chain_present: mem::transmute::<*mut c_void, DXGISwapChainPresentType>(
                hook_present.trampoline(),
            ),
            dxgi_swap_chain_resize_buffers: mem::transmute::<
                *mut c_void,
                DXGISwapChainResizeBuffersType,
            >(hook_resize_buffers.trampoline()),
            d3d12_command_queue_execute_command_lists: mem::transmute::<
                *mut c_void,
                D3D12CommandQueueExecuteCommandListsType,
            >(hook_cqecl.trampoline()),
        });

        Self([hook_present, hook_resize_buffers, hook_cqecl])
    }
}

impl Hooks for ImguiDx12Hooks {
    fn from_render_loop<T>(t: T) -> Box<Self>
    where
        Self: Sized,
        T: ImguiRenderLoop + Send + Sync + 'static,
    {
        Box::new(unsafe { Self::new(t) })
    }

    fn hooks(&self) -> &[MhHook] {
        &self.0
    }

    unsafe fn unhook(&mut self) {
        TRAMPOLINES.take();
        PIPELINE.take().map(|p| p.into_inner().take());
        RENDER_LOOP.take(); // should already be null

        *INITIALIZATION_CONTEXT.lock() = InitializationContext::Empty;
    }
}
