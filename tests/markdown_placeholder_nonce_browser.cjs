// Real-Chromium regression for forged stash placeholders: input that spells a
// renderer placeholder (___ALLOWED_HTML_0___ etc.) inside an attribute must
// stay literal text there. Before the per-render nonce, the restore step
// expanded a stashed block into the attribute and broke out of it. The Node
// pytest harness stubs `document`, so the sanitizer is a no-op there and
// cannot observe the re-assembled attribute.
const { chromium } = require('playwright');

const token = '___ALLOWED_HTML_0___';
// Stash a real block first so the forged token has something to expand into.
const stashed = '[l](https://a.com) ![p https://x.com/x/onerror=alert(1)//](https://n.invalid/p.png)';
const cases = [
  `<img src="https://n.invalid/x.png" alt="${token}">`,
  `<img src="https://n.invalid/x.png" title='${token}'>`,
  `![${token}](https://n.invalid/x.png)`,
  `![a](https://n.invalid/x.png "${token}")`,
  `<a href="https://a.com" title="${token}">t</a>`,
  ...['CODE_BLOCK', 'INLINE_CODE', 'SVG_BLOCK', 'MERMAID_BLOCK', 'MATH_BLOCK']
    .map(kind => `<img src="https://n.invalid/x.png" alt="___${kind}_0___">`),
].map(holder => ({ holder, source: `${stashed}\n\n\`c\`\n\n\`\`\`\nf\n\`\`\`\n\n$$m$$\n\n${holder}` }));

(async () => {
  const origin = process.env.ODYSSEUS_TEST_STATIC_ORIGIN;
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.route('**/nonce-harness', route => route.fulfill({ contentType: 'text/html', body: '<!doctype html><div id="host"></div>' }));
    await page.route('**/api/**', route => route.fulfill({ json: { fonts: {}, value: null } }));
    await page.goto(origin + '/nonce-harness');
    const flagged = await page.evaluate(async (cases) => {
      const markdown = await import('/static/js/markdown.js');
      const host = document.getElementById('host');
      const flagged = [];
      for (const { holder, source } of cases) {
        host.innerHTML = markdown.mdToHtml(source);
        const bad = [];
        for (const el of host.querySelectorAll('*')) {
          for (const attr of el.attributes) {
            const name = attr.name.toLowerCase();
            if (name.startsWith('on') || name === 'style' || !/^[a-z][a-z0-9-]*$/.test(name)) bad.push(`${el.tagName}.${attr.name}`);
          }
        }
        // The forged token must survive as the literal attribute value.
        const literal = [...host.querySelectorAll('img, a')].some(el =>
          [el.getAttribute('alt'), el.getAttribute('title')].some(v => /^___[A-Z_]+_0___$/.test(v || '')));
        if (bad.length || !literal) flagged.push({ holder, bad, literal });
      }
      return flagged;
    }, cases);
    console.log(JSON.stringify({ flagged }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
