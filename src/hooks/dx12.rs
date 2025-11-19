//! Hooks for DirectX 12.
//! 
//! Fixed version - Key improvements:
//! - Command queue selection by num_command_lists (game uses 5+, our test uses 1)
//! - Safe resource cleanup (command queue persists across swap chain changes)
//! - No COM object cloning (store raw pointers)
//! - Proper cleanup order in ResizeBuffers

use std::cell::RefCell;
use std::ffi::c_void;
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use imgui::Context;
use tracing::{debug, error, trace};
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

// State per render thread using thread_local (stable pattern)
struct RenderState {
    swap_chain_ptr: *const c_void,
    swap_chain: Option<IDXGISwapChain3>,
    command_queue_ptr: *mut c_void,  // Store raw pointer, don't clone COM object
    pipeline: Option<Pipeline<D3D12RenderEngine>>,
    initialized: bool,
}

impl RenderState {
    fn new() -> Self {
        Self {
            swap_chain_ptr: std::ptr::null(),
            swap_chain: None,
            command_queue_ptr: std::ptr::null_mut(),
            pipeline: None,
            initialized: false,
        }
    }
    
    fn reset(&mut self) {
        self.initialized = false;
        
        // Clean up pipeline if it exists
        if let Some(mut pipeline) = self.pipeline.take() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pipeline.cleanup();
            }));
            drop(pipeline);
        }
        
        // Clear swap chain references
        self.swap_chain_ptr = std::ptr::null();
        self.swap_chain = None;
        
        // Don't clear command_queue_ptr!
        // Command queues are device-level and persist across swap chain changes
    }
    
    // Validate that swap chain hasn't changed
    fn validate_swap_chain(&self, current_ptr: *const c_void) -> bool {
        self.swap_chain_ptr == current_ptr || self.swap_chain_ptr.is_null()
    }
    
    // Get command queue without cloning
    unsafe fn get_command_queue(&self) -> Option<ID3D12CommandQueue> {
        if self.command_queue_ptr.is_null() {
            None
        } else {
            Some(ID3D12CommandQueue::from_raw(self.command_queue_ptr))
        }
    }
}

thread_local! {
    static RENDER_STATE: RefCell<RenderState> = RefCell::new(RenderState::new());
}

static mut RENDER_LOOP: OnceLock<Box<dyn ImguiRenderLoop + Send + Sync>> = OnceLock::new();
static RENDERING_ACTIVE: AtomicBool = AtomicBool::new(false);

unsafe fn init_pipeline(
    swap_chain: &IDXGISwapChain3,
    command_queue: &ID3D12CommandQueue,
) -> Result<Pipeline<D3D12RenderEngine>> {
    let hwnd = util::try_out_param(|v| swap_chain.GetDesc(v)).map(|desc| desc.OutputWindow)?;

    let mut ctx = Context::create();
    let engine = D3D12RenderEngine::new(command_queue, &mut ctx)?;

    let Some(render_loop) = RENDER_LOOP.take() else {
        error!("Render loop not yet initialized");
        return Err(Error::from_hresult(HRESULT(-1)));
    };

    let pipeline = Pipeline::new(hwnd, ctx, engine, render_loop).map_err(|(e, render_loop)| {
        RENDER_LOOP.get_or_init(move || render_loop);
        e
    })?;

    Ok(pipeline)
}

fn render(swap_chain: &IDXGISwapChain3, swap_chain_ptr: *const c_void) -> Result<()> {
    RENDER_STATE.with(|state_cell| {
        let mut state = state_cell.borrow_mut();
        
        // Validate swap chain
        if !state.validate_swap_chain(swap_chain_ptr) {
            return Ok(());
        }
        
        // Initialize pipeline if needed
        if !state.initialized {
            let command_queue = unsafe { state.get_command_queue() };
            
            if let (Some(sc), Some(cq)) = (&state.swap_chain, command_queue) {
                match unsafe { init_pipeline(sc, &cq) } {
                    Ok(pipeline) => {
                        state.pipeline = Some(pipeline);
                        state.initialized = true;
                    }
                    Err(e) => {
                        error!("Failed to initialize pipeline: {:?}", e);
                        return Err(e);
                    }
                }
            } else {
                return Ok(());
            }
        }
        
        // Render
        if let Some(pipeline) = &mut state.pipeline {
            pipeline.prepare_render()?;
            
            let target: ID3D12Resource = unsafe {
                swap_chain.GetBuffer(swap_chain.GetCurrentBackBufferIndex())?
            };
            
            pipeline.render(target)?;
        }
        
        Ok(())
    })
}

unsafe extern "system" fn dxgi_swap_chain_present_impl(
    swap_chain: IDXGISwapChain3,
    sync_interval: u32,
    flags: u32,
) -> HRESULT {
    let _hook_ejection_guard = HOOK_EJECTION_BARRIER.acquire_ejection_guard();
    
    static mut SKIP_FRAMES: u32 = 120;
    static mut LAST_SWAP_CHAIN_PTR: *const c_void = std::ptr::null();
    
    let current_ptr = swap_chain.as_raw() as *const c_void;
    
    // Handle swap chain changes
    if current_ptr != LAST_SWAP_CHAIN_PTR {
        SKIP_FRAMES = 60;
        RENDERING_ACTIVE.store(false, Ordering::Release);
        
        RENDER_STATE.with(|state_cell| {
            if let Ok(mut state) = state_cell.try_borrow_mut() {
                state.reset();
            }
        });
        
        LAST_SWAP_CHAIN_PTR = current_ptr;
    }
    
    // Skip frames during initialization
    if SKIP_FRAMES > 0 {
        SKIP_FRAMES -= 1;
        
        let Trampolines { dxgi_swap_chain_present, .. } =
            TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");
        return dxgi_swap_chain_present(swap_chain, sync_interval, flags);
    }
    
    // Store swap chain for initialization
    RENDER_STATE.with(|state_cell| {
        if let Ok(mut state) = state_cell.try_borrow_mut() {
            if state.swap_chain.is_none() {
                state.swap_chain = Some(swap_chain.clone());
                state.swap_chain_ptr = current_ptr;
            }
        }
    });

    let Trampolines { dxgi_swap_chain_present, .. } =
        TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");

    // Check if we can render
    let can_render = RENDER_STATE.with(|state_cell| {
        if let Ok(state) = state_cell.try_borrow() {
            state.validate_swap_chain(current_ptr) && 
            (state.initialized || (state.swap_chain.is_some() && !state.command_queue_ptr.is_null()))
        } else {
            false
        }
    });
    
    // Try to render if ready
    if can_render && RENDERING_ACTIVE.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
        let render_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            render(&swap_chain, current_ptr)
        }));
        
        match render_result {
            Ok(Ok(())) => {},
            Ok(Err(e)) => {
                error!("Render error: {e:?}");
            },
            Err(_) => {
                error!("Render panicked!");
                SKIP_FRAMES = 120;
            }
        }
        
        RENDERING_ACTIVE.store(false, Ordering::Release);
    }

    let result = dxgi_swap_chain_present(swap_chain, sync_interval, flags);

    if EJECT_REQUESTED.load(Ordering::SeqCst) {
        perform_eject();
    }

    result
}

unsafe extern "system" fn dxgi_swap_chain_resize_buffers_impl(
    p_this: IDXGISwapChain3,
    buffer_count: u32,
    width: u32,
    height: u32,
    new_format: DXGI_FORMAT,
    flags: u32,
) -> HRESULT {
    let _hook_ejection_guard = HOOK_EJECTION_BARRIER.acquire_ejection_guard();
    
    // Clear references BEFORE calling original
    RENDERING_ACTIVE.store(false, Ordering::Release);
    
    RENDER_STATE.with(|state_cell| {
        if let Ok(mut state) = state_cell.try_borrow_mut() {
            state.initialized = false;
            state.swap_chain = None;
            // Keep command_queue_ptr - it's still valid!
        }
    });
    
    let Trampolines { dxgi_swap_chain_resize_buffers, .. } =
        TRAMPOLINES.get().expect("DirectX 12 trampolines uninitialized");

    let result = dxgi_swap_chain_resize_buffers(p_this, buffer_count, width, height, new_format, flags);
    
    // Clean up pipeline after resize succeeded
    if result.is_ok() {
        RENDER_STATE.with(|state_cell| {
            if let Ok(mut state) = state_cell.try_borrow_mut() {
                if let Some(mut pipeline) = state.pipeline.take() {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        pipeline.cleanup();
                    }));
                }
                state.swap_chain_ptr = std::ptr::null();
            }
        });
    }
    
    result
}

unsafe extern "system" fn d3d12_command_queue_execute_command_lists_impl(
    command_queue: ID3D12CommandQueue,
    num_command_lists: u32,
    command_lists: *mut ID3D12CommandList,
) {
    let _hook_ejection_guard = HOOK_EJECTION_BARRIER.acquire_ejection_guard();
    
    // Capture DIRECT command queues that submit 5+ lists (real game queues)
    // Our test queue only submits 1 list, so we skip it
    RENDER_STATE.with(|state_cell| {
        if let Ok(mut state) = state_cell.try_borrow_mut() {
            let desc = command_queue.GetDesc();
            
            if desc.Type == D3D12_COMMAND_LIST_TYPE_DIRECT && state.command_queue_ptr.is_null() {
                // Game queues typically submit 5+ command lists
                // Our dummy test queue only submits 1
                if num_command_lists >= 5 {
                    let queue_ptr = command_queue.as_raw() as *mut c_void;
                    debug!("Captured command queue (lists: {})", num_command_lists);
                    state.command_queue_ptr = queue_ptr;
                }
            }
        }
    });

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
        }
    };

    let swap_chain: IDXGISwapChain3 = match swap_chain.cast() {
        Ok(v) => v,
        Err(e) => {
            panic!("{e:?}");
        },
    };

    unsafe {
        let sc_vtable = *(swap_chain.as_raw() as *const *const usize);
        let q_vtable = *(command_queue.as_raw() as *const *const usize);
        
        let present_ptr = *sc_vtable.add(8);
        let resize_buffers_ptr = *sc_vtable.add(13);
        let execute_command_lists_ptr = *q_vtable.add(10);
        
        debug!("Vtable addresses extracted:");
        debug!("  Present:              {:p}", present_ptr as *const c_void);
        debug!("  ResizeBuffers:        {:p}", resize_buffers_ptr as *const c_void);
        debug!("  ExecuteCommandLists:  {:p}", execute_command_lists_ptr as *const c_void);
        
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

pub struct ImguiDx12Hooks([MhHook; 3]);

impl ImguiDx12Hooks {
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
        
        // Clean up render state
        RENDER_STATE.with(|state_cell| {
            if let Ok(mut state) = state_cell.try_borrow_mut() {
                state.reset();
            }
        });
        
        RENDER_LOOP.take();
        RENDERING_ACTIVE.store(false, Ordering::Release);
    }
}
