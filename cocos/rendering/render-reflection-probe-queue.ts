/*
 Copyright (c) 2020-2023 Xiamen Yaji Software Co., Ltd.

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

import { SubModel } from '../render-scene/scene/submodel';
import { isEnableEffect, SetIndex } from './define';
import { Device, RenderPass, Shader, CommandBuffer, Texture, SamplerInfo, Address, PipelineState, Attribute, Format, InputAssemblerInfo, BufferInfo, BufferUsageBit, MemoryUsageBit, InputAssembler, Rect, SurfaceTransform, Viewport } from '../gfx';
import { getPhaseID } from './pass-phase';
import { PipelineStateManager } from './pipeline-state-manager';
import { Pass, BatchingSchemes, IMacroPatch } from '../render-scene/core/pass';
import { Model } from '../render-scene/scene/model';
import { ProbeType, ReflectionProbe } from '../render-scene/scene/reflection-probe';
import { Camera, SKYBOX_FLAG } from '../render-scene/scene/camera';
import { PipelineRuntime } from './custom/pipeline';
import { RenderInstancedQueue } from './render-instanced-queue';
import { cclegacy, geometry } from '../core';
import { Material } from '../asset/assets/material';
import { PipelineInputAssemblerData } from './render-pipeline';

const CC_USE_RGBE_OUTPUT = 'CC_USE_RGBE_OUTPUT';
let _phaseID = getPhaseID('default');
let _phaseReflectMapID = getPhaseID('reflect-map');
let _phaseConvertRgbeID = getPhaseID('convert-rgbe');
function getPassIndex (subModel: SubModel): number {
    const passes = subModel.passes;
    const r = cclegacy.rendering;
    if (isEnableEffect()) _phaseID = r.getPhaseID(r.getPassID('default'), 'default');
    for (let k = 0; k < passes.length; k++) {
        if (((!r || !r.enableEffectImport) && passes[k].phase === _phaseID) || (isEnableEffect() && passes[k].phaseID === _phaseID)) {
            return k;
        }
    }
    return -1;
}

function getReflectMapPassIndex (subModel: SubModel): number {
    const passes = subModel.passes;
    const r = cclegacy.rendering;
    if (isEnableEffect()) _phaseReflectMapID = r.getPhaseID(r.getPassID('default'), 'reflect-map');
    for (let k = 0; k < passes.length; k++) {
        if (((!r || !r.enableEffectImport) && passes[k].phase === _phaseReflectMapID)
        || (isEnableEffect() && passes[k].phaseID === _phaseReflectMapID)) {
            return k;
        }
    }
    return -1;
}

function getRGBEPassIndex (subModel: SubModel): number {
    const passes = subModel.passes;
    const r = cclegacy.rendering;
    if (isEnableEffect()) _phaseConvertRgbeID = r.getPhaseID(r.getPassID('default'), 'convert-rgbe');
    for (let k = 0; k < passes.length; k++) {
        if (((!r || !r.enableEffectImport) && passes[k].phase === _phaseConvertRgbeID)
        || (isEnableEffect() && passes[k].phaseID === _phaseConvertRgbeID)) {
            return k;
        }
    }
    return -1;
}
/**
 * @zh
 * 反射探针渲染队列
 */
export class RenderReflectionProbeQueue {
    private _pipeline: PipelineRuntime;
    private _subModelsArray: SubModel[] = [];
    private _passArray: Pass[] = [];
    private _shaderArray: Shader[] = [];
    private _rgbeSubModelsArray: SubModel[] = [];
    private _instancedQueue: RenderInstancedQueue;
    private _patches: IMacroPatch[] = [];

    private _bgMat: Material|null = null;

    public constructor (pipeline: PipelineRuntime) {
        this._pipeline = pipeline;
        this._instancedQueue = new RenderInstancedQueue();
    }
    public gatherRenderObjects (probe: ReflectionProbe, camera: Camera, cmdBuff: CommandBuffer): void {
        this.clear();
        const scene = camera.scene!;
        const sceneData = this._pipeline.pipelineSceneData;
        const skybox = sceneData.skybox;

        if (skybox.enabled && skybox.model && (probe.camera.clearFlag & SKYBOX_FLAG)) {
            this.add(skybox.model);
        }

        const models = scene.models;
        const visibility = probe.visibility;

        for (let i = 0; i < models.length; i++) {
            const model = models[i];
            if (!model.node || scene.isCulledByLod(camera, model)) {
                continue;
            }
            if (((visibility & model.node.layer) !== model.node.layer) && (!(visibility & model.visFlags))) {
                continue;
            }
            if (model.enabled && model.worldBounds && model.bakeToReflectionProbe) {
                if (probe.probeType === ProbeType.CUBE) {
                    if (geometry.intersect.aabbWithAABB(model.worldBounds, probe.boundingBox!)) {
                        this.add(model);
                    }
                } else if (geometry.intersect.aabbFrustum(model.worldBounds, probe.camera.frustum)) {
                    this.add(model);
                }
            }
        }
        this._instancedQueue.uploadBuffers(cmdBuff);
    }

    public clear (): void {
        this._subModelsArray.length = 0;
        this._shaderArray.length = 0;
        this._passArray.length = 0;
        this._instancedQueue.clear();
        this._rgbeSubModelsArray.length = 0;
    }

    public add (model: Model): void {
        const subModels = model.subModels;
        for (let j = 0; j < subModels.length; j++) {
            const subModel = subModels[j];

            //Filter transparent objects
            const isTransparent = subModel.passes[0].blendState.targets[0].blend;
            if (isTransparent) {
                //continue;
            }

            let passIdx = getReflectMapPassIndex(subModel);
            let bUseReflectPass = true;
            if (passIdx < 0) {
                passIdx = getPassIndex(subModel);
                bUseReflectPass = false;
            }
            if (passIdx < 0) { continue; }

            const pass = subModel.passes[passIdx];
            const batchingScheme = pass.batchingScheme;

            // if (!bUseReflectPass) {
            //     this._patches = [];
            //     this._patches = this._patches.concat(subModel.patches!);
            //     const useRGBEPatchs: IMacroPatch[] = [
            //         { name: CC_USE_RGBE_OUTPUT, value: true },
            //     ];
            //     this._patches = this._patches.concat(useRGBEPatchs);
            //     subModel.onMacroPatchesStateChanged(this._patches);
            //     this._rgbeSubModelsArray.push(subModel);
            // }

            if (batchingScheme === BatchingSchemes.INSTANCING) {            // instancing
                const buffer = pass.getInstancedBuffer();
                buffer.merge(subModel, passIdx);
                this._instancedQueue.queue.add(buffer);
            } else {
                const shader = subModel.shaders[passIdx];
                this._subModelsArray.push(subModel);
                if (shader) this._shaderArray.push(shader);
                this._passArray.push(pass);
            }
        }
    }

    /**
     * @zh
     * record CommandBuffer
     */
    public recordCommandBuffer (device: Device, renderPass: RenderPass, cmdBuff: CommandBuffer): void {
        //this._instancedQueue.recordCommandBuffer(device, renderPass, cmdBuff);

        for (let i = 0; i < this._subModelsArray.length; ++i) {
            const subModel = this._subModelsArray[i];
            const shader = this._shaderArray[i];
            const pass = this._passArray[i];
            const ia = subModel.inputAssembler;
            const pso = PipelineStateManager.getOrCreatePipelineState(device, pass, shader, renderPass, ia);
            const descriptorSet = pass.descriptorSet;

            cmdBuff.bindPipelineState(pso);
            cmdBuff.bindDescriptorSet(SetIndex.MATERIAL, descriptorSet);
            cmdBuff.bindDescriptorSet(SetIndex.LOCAL, subModel.descriptorSet);
            cmdBuff.bindInputAssembler(ia);
            cmdBuff.draw(ia);
        }
        //this.resetRGBEMacro();
        //this._instancedQueue.clear();
    }
    // eslint-disable-next-line max-len
    public recordCommandBufferRGBE (camera: Camera, device: Device, renderPass: RenderPass, cmdBuff: CommandBuffer, rgbeTex: Texture| null): void {
        //this._instancedQueue.recordCommandBuffer(device, renderPass, cmdBuff);
        if (!this._bgMat) {
            this._bgMat = new Material();
            this._bgMat.initialize({ effectName: 'util/splash-screen' });
        }
        // const subModel = this._subModelsArray[i];
        // const passIdx = getRGBEPassIndex(subModel);
        // if (passIdx < 0) {
        //     continue;
        // }

        for (let i = 0; i < this._subModelsArray.length; ++i) {
            const subModel = this._subModelsArray[i];
            const rgbePassIdx = getRGBEPassIndex(subModel);
            if (rgbePassIdx < 0) {
                continue;
            }
            const shader = subModel.shaders[rgbePassIdx];
            const pass = subModel.passes[rgbePassIdx];

            // const pass = this._bgMat.passes[0];
            // const shader = pass.getShaderVariant();

            // const handle = pass.getBinding('rgbeTex');
            // pass.bindTexture(handle, rgbeTex!);
            // const samplerInfo = new SamplerInfo();
            // samplerInfo.addressU = Address.CLAMP;
            // samplerInfo.addressV = Address.CLAMP;
            // samplerInfo.addressW = Address.CLAMP;
            // pass.bindSampler(handle, device.getSampler(samplerInfo));

            const ia = this._createQuadInputAssembler();
            const vb = this._genQuadVertexData(device, SurfaceTransform.IDENTITY, new Rect(0, 0, 256, 256));
            ia.quadVB?.update(vb);
            const pso = PipelineStateManager.getOrCreatePipelineState(device, pass, shader, renderPass, ia.quadIA!);
            const descriptorSet = pass.descriptorSet;

            const profilerViewport = new Viewport();
            const profilerScissor = new Rect();
            profilerViewport.width = profilerScissor.width = 256;
            profilerViewport.height = profilerScissor.height = 256;

            cmdBuff.setViewport(profilerViewport);
            cmdBuff.setScissor(profilerScissor);
            cmdBuff.bindPipelineState(pso);
            cmdBuff.bindDescriptorSet(SetIndex.MATERIAL, descriptorSet);
            cmdBuff.bindDescriptorSet(SetIndex.LOCAL, subModel.descriptorSet);
            cmdBuff.bindInputAssembler(ia.quadIA!);
            cmdBuff.draw(ia.quadIA!);
            break;
        }

        // const pass = this._bgMat.passes[0];
        // const shader = pass.getShaderVariant();

        // const handle = pass.getBinding('rgbeTex');
        // pass.bindTexture(handle, rgbeTex!);
        // const samplerInfo = new SamplerInfo();
        // samplerInfo.addressU = Address.CLAMP;
        // samplerInfo.addressV = Address.CLAMP;
        // samplerInfo.addressW = Address.CLAMP;
        // pass.bindSampler(handle, device.getSampler(samplerInfo));

        // const inputAssembler = this._createQuadInputAssembler();
        // const pso = PipelineStateManager.getOrCreatePipelineState(device, pass, shader!, renderPass, inputAssembler);

        // if (pso != null) {
        //     cmdBuff.bindPipelineState(pso);
        //     cmdBuff.bindDescriptorSet(SetIndex.MATERIAL, pass.descriptorSet);
        //     //cmdBuff.bindDescriptorSet(SetIndex.LOCAL, pass.descriptorSet);
        //     cmdBuff.bindInputAssembler(inputAssembler);
        //     cmdBuff.draw(inputAssembler);
        // }
        //this.resetRGBEMacro();
        //this._instancedQueue.clear();
    }

    private _genQuadVertexData (device: Device, surfaceTransform: SurfaceTransform, renderArea: Rect): Float32Array {
        const vbData = new Float32Array(4 * 4);
        const minX = renderArea.x / 256;
        const maxX = (renderArea.x + renderArea.width) / 256;
        let minY = renderArea.y / 256;
        let maxY = (renderArea.y + renderArea.height) / 256;
        if (device.capabilities.screenSpaceSignY > 0) {
            const temp = maxY;
            maxY       = minY;
            minY       = temp;
        }
        let n = 0;
        switch (surfaceTransform) {
        case (SurfaceTransform.IDENTITY):
            n = 0;
            vbData[n++] = -1.0; vbData[n++] = -1.0; vbData[n++] = minX; vbData[n++] = maxY;
            vbData[n++] = 1.0; vbData[n++] = -1.0; vbData[n++] = maxX; vbData[n++] = maxY;
            vbData[n++] = -1.0; vbData[n++] = 1.0; vbData[n++] = minX; vbData[n++] = minY;
            vbData[n++] = 1.0; vbData[n++] = 1.0; vbData[n++] = maxX; vbData[n++] = minY;
            break;
        case (SurfaceTransform.ROTATE_90):
            n = 0;
            vbData[n++] = -1.0; vbData[n++] = -1.0; vbData[n++] = maxX; vbData[n++] = maxY;
            vbData[n++] = 1.0; vbData[n++] = -1.0; vbData[n++] = maxX; vbData[n++] = minY;
            vbData[n++] = -1.0; vbData[n++] = 1.0; vbData[n++] = minX; vbData[n++] = maxY;
            vbData[n++] = 1.0; vbData[n++] = 1.0; vbData[n++] = minX; vbData[n++] = minY;
            break;
        case (SurfaceTransform.ROTATE_180):
            n = 0;
            vbData[n++] = -1.0; vbData[n++] = -1.0; vbData[n++] = minX; vbData[n++] = minY;
            vbData[n++] = 1.0; vbData[n++] = -1.0; vbData[n++] = maxX; vbData[n++] = minY;
            vbData[n++] = -1.0; vbData[n++] = 1.0; vbData[n++] = minX; vbData[n++] = maxY;
            vbData[n++] = 1.0; vbData[n++] = 1.0; vbData[n++] = maxX; vbData[n++] = maxY;
            break;
        case (SurfaceTransform.ROTATE_270):
            n = 0;
            vbData[n++] = -1.0; vbData[n++] = -1.0; vbData[n++] = minX; vbData[n++] = minY;
            vbData[n++] = 1.0; vbData[n++] = -1.0; vbData[n++] = minX; vbData[n++] = maxY;
            vbData[n++] = -1.0; vbData[n++] = 1.0; vbData[n++] = maxX; vbData[n++] = minY;
            vbData[n++] = 1.0; vbData[n++] = 1.0; vbData[n++] = maxX; vbData[n++] = maxY;
            break;
        default:
            break;
        }

        return vbData;
    }

    protected _createQuadInputAssembler (): PipelineInputAssemblerData {
        // create vertex buffer
        const inputAssemblerData = new PipelineInputAssemblerData();

        const vbStride = Float32Array.BYTES_PER_ELEMENT * 4;
        const vbSize = vbStride * 4;
        const device = cclegacy.director.root.device;
        const quadVB = device.createBuffer(new BufferInfo(
            BufferUsageBit.VERTEX | BufferUsageBit.TRANSFER_DST,
            MemoryUsageBit.DEVICE | MemoryUsageBit.HOST,
            vbSize,
            vbStride,
        ));

        if (!quadVB) {
            return inputAssemblerData;
        }

        // create index buffer
        const ibStride = Uint8Array.BYTES_PER_ELEMENT;
        const ibSize = ibStride * 6;

        const quadIB = device.createBuffer(new BufferInfo(
            BufferUsageBit.INDEX | BufferUsageBit.TRANSFER_DST,
            MemoryUsageBit.DEVICE,
            ibSize,
            ibStride,
        ));

        if (!quadIB) {
            return inputAssemblerData;
        }

        const indices = new Uint8Array(6);
        indices[0] = 0; indices[1] = 1; indices[2] = 2;
        indices[3] = 1; indices[4] = 3; indices[5] = 2;

        quadIB.update(indices);

        // create input assembler

        const attributes = new Array<Attribute>(2);
        attributes[0] = new Attribute('a_position', Format.RG32F);
        attributes[1] = new Attribute('a_texCoord', Format.RG32F);

        const quadIA = device.createInputAssembler(new InputAssemblerInfo(
            attributes,
            // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
            [quadVB],
            // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
            quadIB,
        ));

        inputAssemblerData.quadIB = quadIB;
        inputAssemblerData.quadVB = quadVB;
        inputAssemblerData.quadIA = quadIA;
        return inputAssemblerData;
        // const verts = new Float32Array([-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0,  0.5, 0.0]);
        // const vbStride = Float32Array.BYTES_PER_ELEMENT * 4;
        // const vbSize = vbStride * 4;
        // const vertexBuffers = this._pipeline.device.createBuffer(new BufferInfo(
        //     BufferUsageBit.VERTEX | BufferUsageBit.TRANSFER_DST,
        //     MemoryUsageBit.DEVICE,
        //     vbSize,
        //     vbStride,
        // ));
        // vertexBuffers.update(verts);

        // // create index buffer
        // const indices = new Uint16Array([0, 1, 2]);
        // const ibStride = Uint16Array.BYTES_PER_ELEMENT;
        // const ibSize = ibStride * 3;
        // const indicesBuffers = this._pipeline.device.createBuffer(new BufferInfo(
        //     BufferUsageBit.INDEX | BufferUsageBit.TRANSFER_DST,
        //     MemoryUsageBit.DEVICE,
        //     ibSize,
        //     ibStride,
        // ));
        // indicesBuffers.update(indices);

        // const attributes: Attribute[] = [
        //     new Attribute('a_position', Format.RG32F),
        //     new Attribute('a_texCoord', Format.RG32F),
        // ];
        // const IAInfo = new InputAssemblerInfo(attributes, [vertexBuffers], indicesBuffers);
        // return this._pipeline.device.createInputAssembler(IAInfo);
    }
    public resetRGBEMacro (): void {
        for (let i = 0; i < this._rgbeSubModelsArray.length; i++) {
            this._patches = [];
            const subModel = this._rgbeSubModelsArray[i];
            // eslint-disable-next-line prefer-const
            this._patches = this._patches.concat(subModel.patches!);
            if (!this._patches) continue;
            for (let j = 0; j < this._patches.length; j++) {
                const patch = this._patches[j];
                if (patch.name === CC_USE_RGBE_OUTPUT) {
                    this._patches.splice(j, 1);
                    break;
                }
            }
            subModel.onMacroPatchesStateChanged(this._patches);
        }
    }
}
