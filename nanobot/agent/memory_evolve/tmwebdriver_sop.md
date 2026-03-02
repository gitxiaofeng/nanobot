# TMWebDriver SOP

- 禁止import，直接用web_scan/web_execute_js工具。本文件只记录特性和坑。
- 底层：`../TMWebDriver.py`通过Tampermonkey脚本接管用户浏览器（保留登录态/Cookie）
- 非Selenium/Playwright，不需调试浏览器或新数据目录
- 支撑 `web_scan`(只读DOM) / `web_execute_js`(执行JS) 等高层工具

## 限制(isTrusted)
- JS dispatch的事件`isTrusted=false`，敏感操作(文件上传/部分按钮)会被浏览器拦截
- 文件上传：JS无法填充`<input type=file>`，必须ljqCtrl物理点击+Win32轮询文件对话框
  - 流程：SetForegroundWindow→ljqCtrl点上传按钮→FindWindow轮询对话框→输入路径→轮询关闭
- 元素→屏幕物理坐标(ljqCtrl点击前必算)：JS一次取rect+窗口信息，公式：
  - `physX = (screenX + rect中心x) * dpr`，`physY = (screenY + chromeH + rect中心y) * dpr`
  - chromeH = outerHeight - innerHeight，dpr = devicePixelRatio
  - 注意：screenX/Y也是CSS像素，所有值先加后统一乘dpr
- 结论：读信息+普通操作用TMWebDriver；文件上传等敏感操作需配合ljqCtrl

## 导航
- `web_scan` 仅读当前页不导航，切换网站用 `web_execute_js` + `location.href='url'`

## Google图搜
- class名混淆禁硬编码，点击结果用 `[role=button]` div
- web_scan过滤边栏，弹出后用JS：文本`document.body.innerText`，大图遍历img按`naturalWidth`最大取src
- "访问"链接：遍历a找`textContent.includes('访问')`的href
- 缩略图：`img[src^="data:image"]`直接提取；大图src可能截断用`return img.src`

## Chrome下载PDF
场景：PDF链接在浏览器内预览而非下载
```js
fetch('PDF_URL').then(r=>r.blob()).then(b=>{
  const a=document.createElement('a');
  a.href=URL.createObjectURL(b);
  a.download='filename.pdf';
  a.click();
});
```
注意：需同源或CORS允许，跨域先导航到目标域再执行

## Chrome后台标签节流
- 后台标签中`setTimeout`被Chrome intensive throttling延迟到≥1min/次
- TM脚本中detect_newtab的轮询(`setTimeout 150ms × 10`)会超时
- 已修复：移除TM脚本内轮询，改由Python侧`get_session_dict()`前后对比检测新标签
- 同理：TM脚本中任何后台逻辑都应避免依赖setTimeout轮询

## Cookie提取(含HttpOnly)
前提：需先安装`assets/cookie_grabber/`扩展
机制：注入`id="__ljqcg__"`的div→扩展检测后自动将完整cookie写回该元素textContent（含HttpOnly）## 验证码/页面视觉截图
- 优先：JS `canvas.toDataURL()` 直接拿base64（验证码是canvas/img时最干净，无需截屏）
- 备选：`window.open(location.href,'_blank')` 前台开新标签→win32截图→完后close
  - GM_openInTab在web_execute_js不可用（非油猴上下文）
  - 浏览器无JS API切标签页，只能开新的来保证前台

## 跨域iframe操控(postMessage中继)
- 跨域iframe的contentDocument不可访问，web_execute_js只在顶层执行
- TM脚本已改造：iframe内不return，改为监听postMessage并eval执行+回传结果
- 顶层发送：`iframe.contentWindow.postMessage({type:'ljq_exec', id, code}, '*')`
- iframe回传：`{type:'ljq_result', id, result}` 通过window.addEventListener('message')接收
- ⚠只能eval表达式，不支持return/函数体包装，构造代码时注意
- 流程：发postMessage→等→读window._ljqResults[id]获取结果
- 已验证：读取iframe内DOM(document.title)、填写input均成功