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
import { Device, RenderPass, Shader, CommandBuffer, Texture, SamplerInfo, Address, PipelineState, Attribute, Format, InputAssemblerInfo, BufferInfo, BufferUsageBit, MemoryUsageBit, InputAssembler, Rect, SurfaceTransform, Viewport, Filter } from '../gfx';
import { getPhaseID } from './pass-phase';
import { PipelineStateManager } from './pipeline-state-manager';
import { Pass, BatchingSchemes, IMacroPatch } from '../render-scene/core/pass';
import { Model } from '../render-scene/scene/model';
import { ProbeType, ReflectionProbe } from '../render-scene/scene/reflection-probe';
import { Camera, SKYBOX_FLAG } from '../render-scene/scene/camera';
import { PipelineRuntime } from './custom/pipeline';
import { RenderInstancedQueue } from './render-instanced-queue';
import { cclegacy, geometry } from '../core';

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
    private _renderTransparents = false;

    public constructor (pipeline: PipelineRuntime) {
        this._pipeline = pipeline;
        this._instancedQueue = new RenderInstancedQueue();
    }
    public gatherRenderObjects (probe: ReflectionProbe, camera: Camera, cmdBuff: CommandBuffer): void {
        this._renderTransparents = probe.renderTransparents;
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

            //filter transparent objects
            const isTransparent = subModel.passes[0].blendState.targets[0].blend;
            if (isTransparent && !this._renderTransparents) {
                continue;
            }

            let passIdx = getReflectMapPassIndex(subModel);
            let bUseReflectPass = true;
            if (passIdx < 0) {
                passIdx = getPassIndex(subModel);
                bUseReflectPass = false;
            }
            if (passIdx < 0) { continue; }

            if (bUseReflectPass && this._renderTransparents) {
                passIdx = getPassIndex(subModel);
            }

            const pass = subModel.passes[passIdx];
            const batchingScheme = pass.batchingScheme;

            if (!bUseReflectPass && !this._renderTransparents) {
                this._patches = [];
                this._patches = this._patches.concat(subModel.patches!);
                const useRGBEPatchs: IMacroPatch[] = [
                    { name: CC_USE_RGBE_OUTPUT, value: true },
                ];
                this._patches = this._patches.concat(useRGBEPatchs);
                subModel.onMacroPatchesStateChanged(this._patches);
                this._rgbeSubModelsArray.push(subModel);
            }

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
        this._instancedQueue.recordCommandBuffer(device, renderPass, cmdBuff);

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
        if (!this._renderTransparents) {
            this.resetRGBEMacro();
        }
        this._instancedQueue.clear();
    }

    public recordCommandBufferRGBE (device: Device, renderPass: RenderPass, cmdBuff: CommandBuffer, rgbeTex: Texture| null): void {
        for (let i = 0; i < this._subModelsArray.length; ++i) {
            const subModel = this._subModelsArray[i];
            const rgbePassIdx = getRGBEPassIndex(subModel);
            if (rgbePassIdx < 0) {
                continue;
            }
            const shader = subModel.shaders[rgbePassIdx];
            const pass = subModel.passes[rgbePassIdx];

            const isTransparent = pass.blendState.targets[0].blend;

            const handle = pass.getBinding('rgbeTex');
            pass.bindTexture(handle, rgbeTex!);
            pass.bindSampler(handle, device.getSampler(new SamplerInfo(
                Filter.NONE,
                Filter.NONE,
                Filter.NONE,
                Address.CLAMP,
                Address.CLAMP,
                Address.CLAMP,
            )));
            pass.update();

            const ia = this._createQuadInputAssembler();
            const pso = PipelineStateManager.getOrCreatePipelineState(device, pass, shader, renderPass, ia);
            const descriptorSet = pass.descriptorSet;

            cmdBuff.bindPipelineState(pso);
            cmdBuff.bindDescriptorSet(SetIndex.MATERIAL, descriptorSet);
            cmdBuff.bindDescriptorSet(SetIndex.LOCAL, subModel.descriptorSet);
            cmdBuff.bindInputAssembler(ia);
            cmdBuff.draw(ia);
            break;
        }
    }

    protected _createQuadInputAssembler (): InputAssembler {
        const device = cclegacy.director.root.device;
        const minX = 0;
        const maxX = 1;
        let minY = 0;
        let maxY = 1;
        if (device.capabilities.screenSpaceSignY > 0) {
            const temp = maxY;
            maxY       = minY;
            minY       = temp;
        }
        const verts = new Float32Array([-1, -1, minX, maxY, 1, -1, maxX, maxY, -1, 1, minX, minY, 1, 1, maxX, minY]);
        const vbStride = Float32Array.BYTES_PER_ELEMENT * 4;
        const vbSize = vbStride * 4;
        const vertexBuffers = this._pipeline.device.createBuffer(new BufferInfo(
            BufferUsageBit.VERTEX | BufferUsageBit.TRANSFER_DST,
            MemoryUsageBit.DEVICE,
            vbSize,
            vbStride,
        ));
        vertexBuffers.update(verts);

        // create index buffer
        const indices = new Uint16Array([0, 1, 2, 1, 3, 2]);
        const ibStride = Uint16Array.BYTES_PER_ELEMENT;
        const ibSize = ibStride * 6;
        const indicesBuffers = this._pipeline.device.createBuffer(new BufferInfo(
            BufferUsageBit.INDEX | BufferUsageBit.TRANSFER_DST,
            MemoryUsageBit.DEVICE,
            ibSize,
            ibStride,
        ));
        indicesBuffers.update(indices);

        const attributes: Attribute[] = [
            new Attribute('a_position', Format.RG32F),
            new Attribute('a_texCoord', Format.RG32F),
        ];
        const IAInfo = new InputAssemblerInfo(attributes, [vertexBuffers], indicesBuffers);
        return this._pipeline.device.createInputAssembler(IAInfo);
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
