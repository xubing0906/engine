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

/**
 * ========================= !DO NOT CHANGE THE FOLLOWING SECTION MANUALLY! =========================
 * The following section is auto-generated.
 * ========================= !DO NOT CHANGE THE FOLLOWING SECTION MANUALLY! =========================
 */
// clang-format off
#include "FGDispatcherTypes.h"

namespace cc {

namespace render {

ResourceAccessGraph::ResourceAccessGraph(const allocator_type& alloc) noexcept
: _vertices(alloc),
  passID(alloc),
  access(alloc),
  passIndex(alloc),
  resourceNames(alloc),
  resourceIndex(alloc),
  leafPasses(alloc),
  culledPasses(alloc),
  accessRecord(alloc),
  resourceLifeRecord(alloc),
  topologicalOrder(alloc) {}

// ContinuousContainer
void ResourceAccessGraph::reserve(vertices_size_type sz) {
    _vertices.reserve(sz);
    passID.reserve(sz);
    access.reserve(sz);
}

ResourceAccessGraph::Vertex::Vertex(const allocator_type& alloc) noexcept
: outEdges(alloc),
  inEdges(alloc) {}

ResourceAccessGraph::Vertex::Vertex(Vertex&& rhs, const allocator_type& alloc)
: outEdges(std::move(rhs.outEdges), alloc),
  inEdges(std::move(rhs.inEdges), alloc) {}

ResourceAccessGraph::Vertex::Vertex(Vertex const& rhs, const allocator_type& alloc)
: outEdges(rhs.outEdges, alloc),
  inEdges(rhs.inEdges, alloc) {}

RelationGraph::RelationGraph(const allocator_type& alloc) noexcept
: _vertices(alloc),
  descID(alloc),
  vertexMap(alloc) {}

// ContinuousContainer
void RelationGraph::reserve(vertices_size_type sz) {
    _vertices.reserve(sz);
    descID.reserve(sz);
}

RelationGraph::Vertex::Vertex(const allocator_type& alloc) noexcept
: outEdges(alloc),
  inEdges(alloc) {}

RelationGraph::Vertex::Vertex(Vertex&& rhs, const allocator_type& alloc)
: outEdges(std::move(rhs.outEdges), alloc),
  inEdges(std::move(rhs.inEdges), alloc) {}

RelationGraph::Vertex::Vertex(Vertex const& rhs, const allocator_type& alloc)
: outEdges(rhs.outEdges, alloc),
  inEdges(rhs.inEdges, alloc) {}

FrameGraphDispatcher::FrameGraphDispatcher(ResourceGraph& resourceGraphIn, const RenderGraph& graphIn, LayoutGraphData& layoutGraphIn, boost::container::pmr::memory_resource* scratchIn, const allocator_type& alloc) noexcept
: resourceAccessGraph(alloc),
  resourceGraph(resourceGraphIn),
  graph(graphIn),
  layoutGraph(layoutGraphIn),
  scratch(scratchIn),
  externalResMap(alloc),
  relationGraph(alloc) {}

} // namespace render

} // namespace cc

// clang-format on
