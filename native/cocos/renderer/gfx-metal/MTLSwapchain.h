/****************************************************************************
 Copyright (c) 2021-2022 Xiamen Yaji Software Co., Ltd.

 http://www.cocos.com

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated engine source code (the "Software"), a limited,
 worldwide, royalty-free, non-assignable, revocable and non-exclusive license
 to use Cocos Creator solely to develop games on your target platforms. You shall
 not use Cocos Creator software for developing other software or tools that's
 used for developing games. You are not granted to publish, distribute,
 sublicense, and/or sell copies of Cocos Creator.

 The software or tools in this License Agreement are licensed, not sold.
 Xiamen Yaji Software Co., Ltd. reserves all rights not expressly granted to you.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
****************************************************************************/

#pragma once
#import <QuartzCore/CAMetalLayer.h>
#import "MTLTexture.h"
#import "gfx-base/GFXSwapchain.h"

namespace cc {
namespace gfx {

struct CCMTLGPUSwapChainObject;
class CCMTLTexture;

class CCMTLSwapchain final : public Swapchain {
public:
    CCMTLSwapchain();
    ~CCMTLSwapchain();

    inline CCMTLGPUSwapChainObject *gpuSwapChainObj() { return _gpuSwapchainObj; }
    CCMTLTexture *colorTexture();
    CCMTLTexture *depthStencilTexture();

    id<CAMetalDrawable> currentDrawable();

    void acquire();

    void release();

protected:
    void doInit(const SwapchainInfo &info) override;
    void doDestroy() override;
    void doResize(uint32_t width, uint32_t height, SurfaceTransform transform) override;
    void doDestroySurface() override;
    void doCreateSurface(void *windowHandle) override;

private:
    CCMTLGPUSwapChainObject *_gpuSwapchainObj = nullptr;
};

} // namespace gfx
} // namespace cc
