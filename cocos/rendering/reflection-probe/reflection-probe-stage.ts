/*
 Copyright (c) 2022-2023 Xiamen Yaji Software Co., Ltd.

 https://www.cocos.com/

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

import { ccclass } from 'cc.decorator';
import { Color, Rect, Framebuffer, ClearFlagBit, Texture, TextureBlit, Filter, TextureInfo, TextureType, TextureUsageBit, Format, ColorAttachment, LoadOp, StoreOp, DepthStencilAttachment, RenderPass, RenderPassInfo, FramebufferInfo } from '../../gfx';
import { IRenderStageInfo, RenderStage } from '../render-stage';
import { ForwardStagePriority } from '../enum';
import { ForwardPipeline } from '../forward/forward-pipeline';
import { SetIndex } from '../define';
import { ReflectionProbeFlow } from './reflection-probe-flow';
import { Camera, CameraProjection, ReflectionProbe, ToneMappingType } from '../../render-scene/scene';
import { RenderReflectionProbeQueue } from '../render-reflection-probe-queue';
import { Vec3 } from '../../core';
import { packRGBE } from '../../core/math/color';

const colors: Color[] = [new Color(1, 1, 1, 1)];

/**
 * @en reflection probe render stage
 * @zh 反射探针渲染阶段。
 */
@ccclass('ReflectionProbeStage')
export class ReflectionProbeStage extends RenderStage {
    /**
     * @en A common initialization info for reflection probe render stage
     * @zh 一个通用的 reflection probe stage 的初始化信息对象
     */
    public static initInfo: IRenderStageInfo = {
        name: 'ReflectionProbeStage',
        priority: ForwardStagePriority.FORWARD,
        tag: 0,
    };

    private _frameBuffer: Framebuffer | null = null;
    private _renderArea = new Rect();
    private _probe: ReflectionProbe | null = null;
    private _probeRenderQueue!: RenderReflectionProbeQueue;
    private _rgbeColor = new Vec3();
    private _rgbeFramebuffer: Framebuffer | null = null;
    private _rgbeRenderPass: RenderPass|null = null;

    public initialize (info: IRenderStageInfo): boolean {
        super.initialize(info);
        return true;
    }

    /**
     * @en Sets the probe info
     * @zh 设置probe信息
     * @param probe
     * @param frameBuffer
     */
    public setUsageInfo (probe: ReflectionProbe, frameBuffer: Framebuffer): void {
        this._probe = probe;
        this._frameBuffer = frameBuffer;
        this._initRGBEFrameBuffer();
    }

    public destroy (): void {
        this._frameBuffer = null;
        this._probeRenderQueue?.clear();
        this._rgbeFramebuffer = null;
        this._rgbeRenderPass = null;
    }

    public clearFramebuffer (camera: Camera): void {
        if (!this._frameBuffer) { return; }

        const pipeline = this._pipeline as ForwardPipeline;
        const pipelineSceneData = pipeline.pipelineSceneData;
        const shadingScale = pipelineSceneData.shadingScale;
        const vp = camera.viewport;
        const size = this._probe!.resolution;
        this._renderArea.x = vp.x * size;
        this._renderArea.y = vp.y * size;
        this._renderArea.width = vp.width * size * shadingScale;
        this._renderArea.height = vp.height * size * shadingScale;
        const cmdBuff = pipeline.commandBuffers[0];
        const renderPass = this._frameBuffer.renderPass;

        cmdBuff.beginRenderPass(
            renderPass,
            this._frameBuffer,
            this._renderArea,
            colors,
            camera.clearDepth,
            camera.clearStencil,
        );
        cmdBuff.endRenderPass();
    }

    public render (camera: Camera): void {
        const pipeline = this._pipeline;
        const cmdBuff = pipeline.commandBuffers[0];
        this._probeRenderQueue.gatherRenderObjects(this._probe!, camera, cmdBuff);
        pipeline.pipelineUBO.updateCameraUBO(this._probe!.camera);

        this._renderArea.x = 0;
        this._renderArea.y = 0;
        this._renderArea.width = this._probe!.renderArea().x;
        this._renderArea.height = this._probe!.renderArea().y;

        let buffer = this._frameBuffer;
        if (this._probe!.renderTransparents) {
            buffer =  this._rgbeFramebuffer;
        }
        const renderPass = buffer!.renderPass;

        if (this._probe!.camera.clearFlag & ClearFlagBit.COLOR) {
            this._rgbeColor.x = this._probe!.camera.clearColor.x;
            this._rgbeColor.y = this._probe!.camera.clearColor.y;
            this._rgbeColor.z = this._probe!.camera.clearColor.z;
            const rgbe = packRGBE(this._rgbeColor);
            colors[0].x = rgbe.x;
            colors[0].y = rgbe.y;
            colors[0].z = rgbe.z;
            colors[0].w = rgbe.w;
        }
        const device = pipeline.device;
        cmdBuff.beginRenderPass(
            renderPass,
            buffer!,
            this._renderArea,
            colors,
            this._probe!.camera.clearDepth,
            this._probe!.camera.clearStencil,
        );
        cmdBuff.bindDescriptorSet(SetIndex.GLOBAL, pipeline.descriptorSet);

        this._probeRenderQueue.recordCommandBuffer(device, renderPass, cmdBuff);
        cmdBuff.endRenderPass();

        //convert to rgbe
        if (this._probe!.renderTransparents) {
            this._renderRGBE(camera);
        }

        pipeline.pipelineUBO.updateCameraUBO(camera);
    }

    private _renderRGBE (camera: Camera): void {
        const pipeline = this._pipeline;
        const cmdBuff = pipeline.commandBuffers[0];

        this._renderArea.x = 0;
        this._renderArea.y = 0;
        this._renderArea.width = this._probe!.renderArea().x;
        this._renderArea.height = this._probe!.renderArea().y;

        const renderPass = this._frameBuffer!.renderPass;

        if (this._probe!.camera.clearFlag & ClearFlagBit.COLOR) {
            this._rgbeColor.x = this._probe!.camera.clearColor.x;
            this._rgbeColor.y = this._probe!.camera.clearColor.y;
            this._rgbeColor.z = this._probe!.camera.clearColor.z;
            const rgbe = packRGBE(this._rgbeColor);
            colors[0].x = rgbe.x;
            colors[0].y = rgbe.y;
            colors[0].z = rgbe.z;
            colors[0].w = rgbe.w;
        }
        const device = pipeline.device;

        this._probe!.setProjectionType(CameraProjection.ORTHO);
        this._probe!.resetCameraTransform();
        pipeline.pipelineUBO.updateCameraUBO(this._probe!.camera);

        cmdBuff.beginRenderPass(
            renderPass,
            this._frameBuffer!,
            this._renderArea,
            colors,
            camera.clearDepth,
            camera.clearStencil,
        );
        cmdBuff.bindDescriptorSet(SetIndex.GLOBAL, pipeline.descriptorSet);
        this._probeRenderQueue.recordCommandBufferRGBE(device, renderPass, cmdBuff, this._rgbeFramebuffer!.colorTextures[0]);
        cmdBuff.endRenderPass();

        this._probe!.setProjectionType(CameraProjection.PERSPECTIVE);
    }

    public activate (pipeline: ForwardPipeline, flow: ReflectionProbeFlow): void {
        super.activate(pipeline, flow);
        this._probeRenderQueue = new RenderReflectionProbeQueue(pipeline);
    }

    private _initRGBEFrameBuffer (): void {
        const pipeline = this._pipeline;
        const device = pipeline.device;
        const format = Format.RGBA32F;

        if (!this._rgbeRenderPass) {
            const colorAttachment = new ColorAttachment();
            colorAttachment.format = format;
            colorAttachment.loadOp = LoadOp.CLEAR;
            colorAttachment.storeOp = StoreOp.STORE;
            colorAttachment.sampleCount = 1;

            const depthStencilAttachment = new DepthStencilAttachment();
            depthStencilAttachment.format = Format.DEPTH_STENCIL;
            depthStencilAttachment.depthLoadOp = LoadOp.CLEAR;
            depthStencilAttachment.depthStoreOp = StoreOp.DISCARD;
            depthStencilAttachment.stencilLoadOp = LoadOp.CLEAR;
            depthStencilAttachment.stencilStoreOp = StoreOp.DISCARD;
            depthStencilAttachment.sampleCount = 1;

            const renderPassInfo = new RenderPassInfo([colorAttachment], depthStencilAttachment);
            this._rgbeRenderPass = device.createRenderPass(renderPassInfo);

            const renderTargets: Texture[] = [];
            renderTargets.push(device.createTexture(new TextureInfo(
                TextureType.TEX2D,
                TextureUsageBit.COLOR_ATTACHMENT | TextureUsageBit.SAMPLED,
                format,
                this._probe!.renderArea().x,
                this._probe!.renderArea().y,
            )));

            const depth = device.createTexture(new TextureInfo(
                TextureType.TEX2D,
                TextureUsageBit.DEPTH_STENCIL_ATTACHMENT,
                Format.DEPTH_STENCIL,
                this._probe!.renderArea().x,
                this._probe!.renderArea().y,
            ));

            this._rgbeFramebuffer = device.createFramebuffer(new FramebufferInfo(
                this._rgbeRenderPass,
                renderTargets,
                depth,
            ));
        }
    }
}
