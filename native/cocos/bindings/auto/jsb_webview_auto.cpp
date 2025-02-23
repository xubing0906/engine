// clang-format off

/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (https://www.swig.org).
 * Version 4.1.0
 *
 * Do not make changes to this file unless you know what you are doing - modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

/****************************************************************************
 Copyright (c) 2022 Xiamen Yaji Software Co., Ltd.

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

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4101)
#endif


#define SWIG_STD_MOVE(OBJ) std::move(OBJ)


#include <stdio.h>


#include "bindings/jswrapper/SeApi.h"
#include "bindings/manual/jsb_conversions.h"
#include "bindings/manual/jsb_global.h"


#include "bindings/auto/jsb_webview_auto.h"



se::Class* __jsb_cc_WebView_class = nullptr;
se::Object* __jsb_cc_WebView_proto = nullptr;
SE_DECLARE_FINALIZE_FUNC(js_delete_cc_WebView) 

static bool js_cc_WebView_create_static(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *result = 0 ;
    
    if(argc != 0) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 0);
        return false;
    }
    result = (cc::WebView *)cc::WebView::create();
    
    ok &= nativevalue_to_se(result, s.rval(), s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    SE_HOLD_RETURN_VALUE(result, s.thisObject(), s.rval()); 
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_create_static) 

static bool js_cc_WebView_destroy(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    
    if(argc != 0) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 0);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    (arg1)->destroy();
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_destroy) 

static bool js_cc_WebView_setJavascriptInterfaceScheme(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    ccstd::string *arg2 = 0 ;
    ccstd::string temp2 ;
    
    if(argc != 1) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 1);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &temp2, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg2 = &temp2;
    
    (arg1)->setJavascriptInterfaceScheme((ccstd::string const &)*arg2);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_setJavascriptInterfaceScheme) 

static bool js_cc_WebView_loadData(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    cc::Data *arg2 = 0 ;
    ccstd::string *arg3 = 0 ;
    ccstd::string *arg4 = 0 ;
    ccstd::string *arg5 = 0 ;
    cc::Data temp2 ;
    ccstd::string temp3 ;
    ccstd::string temp4 ;
    ccstd::string temp5 ;
    
    if(argc != 4) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 4);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &temp2, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg2 = &temp2;
    
    
    ok &= sevalue_to_native(args[1], &temp3, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg3 = &temp3;
    
    
    ok &= sevalue_to_native(args[2], &temp4, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg4 = &temp4;
    
    
    ok &= sevalue_to_native(args[3], &temp5, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg5 = &temp5;
    
    (arg1)->loadData((cc::Data const &)*arg2,(ccstd::string const &)*arg3,(ccstd::string const &)*arg4,(ccstd::string const &)*arg5);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_loadData) 

static bool js_cc_WebView_loadHTMLString__SWIG_0(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    ccstd::string *arg2 = 0 ;
    ccstd::string *arg3 = 0 ;
    ccstd::string temp2 ;
    ccstd::string temp3 ;
    
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &temp2, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg2 = &temp2;
    
    
    ok &= sevalue_to_native(args[1], &temp3, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg3 = &temp3;
    
    (arg1)->loadHTMLString((ccstd::string const &)*arg2,(ccstd::string const &)*arg3);
    
    
    return true;
}

static bool js_cc_WebView_loadHTMLString__SWIG_1(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    ccstd::string *arg2 = 0 ;
    ccstd::string temp2 ;
    
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &temp2, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg2 = &temp2;
    
    (arg1)->loadHTMLString((ccstd::string const &)*arg2);
    
    
    return true;
}

static bool js_cc_WebView_loadHTMLString(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    
    
    if (argc == 2) {
        ok = js_cc_WebView_loadHTMLString__SWIG_0(s);
        if (ok) {
            return true; 
        }
    } 
    if (argc == 1) {
        ok = js_cc_WebView_loadHTMLString__SWIG_1(s);
        if (ok) {
            return true; 
        }
    } 
    SE_REPORT_ERROR("wrong number of arguments: %d", (int)argc);
    return false;
}
SE_BIND_FUNC(js_cc_WebView_loadHTMLString) 

static bool js_cc_WebView_loadURL(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    ccstd::string *arg2 = 0 ;
    ccstd::string temp2 ;
    
    if(argc != 1) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 1);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &temp2, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg2 = &temp2;
    
    (arg1)->loadURL((ccstd::string const &)*arg2);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_loadURL) 

static bool js_cc_WebView_loadFile(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    ccstd::string *arg2 = 0 ;
    ccstd::string temp2 ;
    
    if(argc != 1) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 1);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &temp2, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg2 = &temp2;
    
    (arg1)->loadFile((ccstd::string const &)*arg2);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_loadFile) 

static bool js_cc_WebView_stopLoading(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    
    if(argc != 0) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 0);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    (arg1)->stopLoading();
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_stopLoading) 

static bool js_cc_WebView_reload(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    
    if(argc != 0) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 0);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    (arg1)->reload();
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_reload) 

static bool js_cc_WebView_canGoBack(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    bool result;
    
    if(argc != 0) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 0);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    result = (bool)(arg1)->canGoBack();
    
    ok &= nativevalue_to_se(result, s.rval(), s.thisObject());
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_canGoBack) 

static bool js_cc_WebView_canGoForward(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    bool result;
    
    if(argc != 0) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 0);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    result = (bool)(arg1)->canGoForward();
    
    ok &= nativevalue_to_se(result, s.rval(), s.thisObject());
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_canGoForward) 

static bool js_cc_WebView_goBack(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    
    if(argc != 0) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 0);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    (arg1)->goBack();
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_goBack) 

static bool js_cc_WebView_goForward(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    
    if(argc != 0) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 0);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    (arg1)->goForward();
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_goForward) 

static bool js_cc_WebView_evaluateJS(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    ccstd::string *arg2 = 0 ;
    ccstd::string temp2 ;
    
    if(argc != 1) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 1);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &temp2, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg2 = &temp2;
    
    (arg1)->evaluateJS((ccstd::string const &)*arg2);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_evaluateJS) 

static bool js_cc_WebView_setScalesPageToFit(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    bool arg2 ;
    
    if(argc != 1) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 1);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &arg2);
    SE_PRECONDITION2(ok, false, "Error processing arguments"); 
    (arg1)->setScalesPageToFit(arg2);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_setScalesPageToFit) 

static bool js_cc_WebView_setOnShouldStartLoading(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    std::function< bool (cc::WebView *,ccstd::string const &) > *arg2 = 0 ;
    std::function< bool (cc::WebView *,ccstd::string const &) > temp2 ;
    
    if(argc != 1) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 1);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &temp2, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg2 = &temp2;
    
    (arg1)->setOnShouldStartLoading((std::function< bool (cc::WebView *,ccstd::string const &) > const &)*arg2);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_setOnShouldStartLoading) 

static bool js_cc_WebView_setOnDidFinishLoading(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    cc::WebView::ccWebViewCallback *arg2 = 0 ;
    cc::WebView::ccWebViewCallback temp2 ;
    
    if(argc != 1) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 1);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &temp2, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg2 = &temp2;
    
    (arg1)->setOnDidFinishLoading((cc::WebView::ccWebViewCallback const &)*arg2);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_setOnDidFinishLoading) 

static bool js_cc_WebView_setOnDidFailLoading(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    cc::WebView::ccWebViewCallback *arg2 = 0 ;
    cc::WebView::ccWebViewCallback temp2 ;
    
    if(argc != 1) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 1);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &temp2, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg2 = &temp2;
    
    (arg1)->setOnDidFailLoading((cc::WebView::ccWebViewCallback const &)*arg2);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_setOnDidFailLoading) 

static bool js_cc_WebView_setOnJSCallback(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    cc::WebView::ccWebViewCallback *arg2 = 0 ;
    cc::WebView::ccWebViewCallback temp2 ;
    
    if(argc != 1) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 1);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &temp2, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    arg2 = &temp2;
    
    (arg1)->setOnJSCallback((cc::WebView::ccWebViewCallback const &)*arg2);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_setOnJSCallback) 

static bool js_cc_WebView_getOnShouldStartLoading(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    std::function< bool (cc::WebView *,ccstd::string const &) > result;
    
    if(argc != 0) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 0);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    result = ((cc::WebView const *)arg1)->getOnShouldStartLoading();
    
    ok &= nativevalue_to_se(result, s.rval(), s.thisObject() /*ctx*/);
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    SE_HOLD_RETURN_VALUE(result, s.thisObject(), s.rval());
    
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_getOnShouldStartLoading) 

static bool js_cc_WebView_getOnDidFinishLoading(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    cc::WebView::ccWebViewCallback result;
    
    if(argc != 0) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 0);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    result = ((cc::WebView const *)arg1)->getOnDidFinishLoading();
    
    ok &= nativevalue_to_se(result, s.rval(), s.thisObject() /*ctx*/);
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    SE_HOLD_RETURN_VALUE(result, s.thisObject(), s.rval());
    
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_getOnDidFinishLoading) 

static bool js_cc_WebView_getOnDidFailLoading(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    cc::WebView::ccWebViewCallback result;
    
    if(argc != 0) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 0);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    result = ((cc::WebView const *)arg1)->getOnDidFailLoading();
    
    ok &= nativevalue_to_se(result, s.rval(), s.thisObject() /*ctx*/);
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    SE_HOLD_RETURN_VALUE(result, s.thisObject(), s.rval());
    
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_getOnDidFailLoading) 

static bool js_cc_WebView_getOnJSCallback(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    cc::WebView::ccWebViewCallback result;
    
    if(argc != 0) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 0);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    result = ((cc::WebView const *)arg1)->getOnJSCallback();
    
    ok &= nativevalue_to_se(result, s.rval(), s.thisObject() /*ctx*/);
    SE_PRECONDITION2(ok, false, "Error processing arguments");
    SE_HOLD_RETURN_VALUE(result, s.thisObject(), s.rval());
    
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_getOnJSCallback) 

static bool js_cc_WebView_setBounces(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    bool arg2 ;
    
    if(argc != 1) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 1);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &arg2);
    SE_PRECONDITION2(ok, false, "Error processing arguments"); 
    (arg1)->setBounces(arg2);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_setBounces) 

static bool js_cc_WebView_setVisible(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    bool arg2 ;
    
    if(argc != 1) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 1);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &arg2);
    SE_PRECONDITION2(ok, false, "Error processing arguments"); 
    (arg1)->setVisible(arg2);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_setVisible) 

static bool js_cc_WebView_setFrame(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    float arg2 ;
    float arg3 ;
    float arg4 ;
    float arg5 ;
    
    if(argc != 4) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 4);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &arg2, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments"); 
    
    ok &= sevalue_to_native(args[1], &arg3, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments"); 
    
    ok &= sevalue_to_native(args[2], &arg4, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments"); 
    
    ok &= sevalue_to_native(args[3], &arg5, s.thisObject());
    SE_PRECONDITION2(ok, false, "Error processing arguments"); 
    (arg1)->setFrame(arg2,arg3,arg4,arg5);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_setFrame) 

static bool js_cc_WebView_setBackgroundTransparent(se::State& s)
{
    CC_UNUSED bool ok = true;
    const auto& args = s.args();
    size_t argc = args.size();
    cc::WebView *arg1 = (cc::WebView *) NULL ;
    bool arg2 ;
    
    if(argc != 1) {
        SE_REPORT_ERROR("wrong number of arguments: %d, was expecting %d", (int)argc, 1);
        return false;
    }
    arg1 = SE_THIS_OBJECT<cc::WebView>(s);
    if (nullptr == arg1) return true;
    
    ok &= sevalue_to_native(args[0], &arg2);
    SE_PRECONDITION2(ok, false, "Error processing arguments"); 
    (arg1)->setBackgroundTransparent(arg2);
    
    
    return true;
}
SE_BIND_FUNC(js_cc_WebView_setBackgroundTransparent) 

bool js_register_cc_WebView(se::Object* obj) {
    auto* cls = se::Class::create("WebView", obj, nullptr, nullptr); 
    
    
    cls->defineFunction("destroy", _SE(js_cc_WebView_destroy)); 
    cls->defineFunction("setJavascriptInterfaceScheme", _SE(js_cc_WebView_setJavascriptInterfaceScheme)); 
    cls->defineFunction("loadData", _SE(js_cc_WebView_loadData)); 
    cls->defineFunction("loadHTMLString", _SE(js_cc_WebView_loadHTMLString)); 
    cls->defineFunction("loadURL", _SE(js_cc_WebView_loadURL)); 
    cls->defineFunction("loadFile", _SE(js_cc_WebView_loadFile)); 
    cls->defineFunction("stopLoading", _SE(js_cc_WebView_stopLoading)); 
    cls->defineFunction("reload", _SE(js_cc_WebView_reload)); 
    cls->defineFunction("canGoBack", _SE(js_cc_WebView_canGoBack)); 
    cls->defineFunction("canGoForward", _SE(js_cc_WebView_canGoForward)); 
    cls->defineFunction("goBack", _SE(js_cc_WebView_goBack)); 
    cls->defineFunction("goForward", _SE(js_cc_WebView_goForward)); 
    cls->defineFunction("evaluateJS", _SE(js_cc_WebView_evaluateJS)); 
    cls->defineFunction("setScalesPageToFit", _SE(js_cc_WebView_setScalesPageToFit)); 
    cls->defineFunction("setOnShouldStartLoading", _SE(js_cc_WebView_setOnShouldStartLoading)); 
    cls->defineFunction("setOnDidFinishLoading", _SE(js_cc_WebView_setOnDidFinishLoading)); 
    cls->defineFunction("setOnDidFailLoading", _SE(js_cc_WebView_setOnDidFailLoading)); 
    cls->defineFunction("setOnJSCallback", _SE(js_cc_WebView_setOnJSCallback)); 
    cls->defineFunction("getOnShouldStartLoading", _SE(js_cc_WebView_getOnShouldStartLoading)); 
    cls->defineFunction("getOnDidFinishLoading", _SE(js_cc_WebView_getOnDidFinishLoading)); 
    cls->defineFunction("getOnDidFailLoading", _SE(js_cc_WebView_getOnDidFailLoading)); 
    cls->defineFunction("getOnJSCallback", _SE(js_cc_WebView_getOnJSCallback)); 
    cls->defineFunction("setBounces", _SE(js_cc_WebView_setBounces)); 
    cls->defineFunction("setVisible", _SE(js_cc_WebView_setVisible)); 
    cls->defineFunction("setFrame", _SE(js_cc_WebView_setFrame)); 
    cls->defineFunction("setBackgroundTransparent", _SE(js_cc_WebView_setBackgroundTransparent)); 
    
    
    cls->defineStaticFunction("create", _SE(js_cc_WebView_create_static)); 
    
    
    cls->install();
    JSBClassType::registerClass<cc::WebView>(cls);
    
    __jsb_cc_WebView_proto = cls->getProto();
    __jsb_cc_WebView_class = cls;
    se::ScriptEngine::getInstance()->clearException();
    return true;
}




bool register_all_webview(se::Object* obj) {
    // Get the ns
    se::Value nsVal;
    if (!obj->getProperty("jsb", &nsVal, true))
    {
        se::HandleObject jsobj(se::Object::createPlainObject());
        nsVal.setObject(jsobj);
        obj->setProperty("jsb", nsVal);
    }
    se::Object* ns = nsVal.toObject();
    /* Register classes */
    js_register_cc_WebView(ns); 
    
    /* Register global variables & global functions */
    
    
    
    return true;
}


#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
// clang-format on
