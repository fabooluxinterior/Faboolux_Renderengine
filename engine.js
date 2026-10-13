async function initWebGPU() {
    if (!navigator.gpu) {
        throw new Error("WebGPU is not supported on this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
        throw new Error("No appropriate GPU adapter found.");
    }

    const device = await adapter.requestDevice();
    const canvas = document.getElementById("engineCanvas");
    const context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();

    context.configure({
        device: device,
        format: format,
        alphaMode: "premultiplied",
    });

    return { device, context, format };
}

const shaders = `
@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, -0.5)
    );
    return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
}

@fragment
fn fragmentMain() -> @location(0) vec4<f32> {
    // Orange base color output
    return vec4<f32>(0.8, 0.4, 0.1, 1.0); 
}
`;

async function startEngine() {
    const { device, context, format } = await initWebGPU();

    const shaderModule = device.createShaderModule({ code: shaders });

    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: shaderModule,
            entryPoint: "vertexMain",
        },
        fragment: {
            module: shaderModule,
            entryPoint: "fragmentMain",
            targets: [{ format: format }],
        },
        primitive: {
            topology: "triangle-list",
        },
    });

    function renderFrame() {
        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

        const renderPassDescriptor = {
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }, // Dark gray background
                loadOp: "clear",
                storeOp: "store",
            }],
        };

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(pipeline);
        passEncoder.draw(3); 
        passEncoder.end();

        device.queue.submit([commandEncoder.finish()]);
        
        requestAnimationFrame(renderFrame);
    }

    requestAnimationFrame(renderFrame);
}

startEngine().catch(console.error);