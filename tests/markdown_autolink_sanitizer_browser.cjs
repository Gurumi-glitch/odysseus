// Real-Chromium regression for the autolink passes: every <a>/<img> the
// renderer or the model produced must reach the DOM without an attribute the
// autolink spliced in (onerror=, style=, junk attrs) and without nested <a>.
// The Node pytest harness stubs `document`, so sanitizeAllowedHtml is a no-op
// there and cannot catch markup that is re-assembled after sanitization.
const { chromium } = require('playwright');

const cases = [
  // Renderer-generated markup: URL inside image alt/title or link label.
  '![x https://x.com/x/onerror=alert(1)//](https://n.invalid/x.png)',
  '![看https://x.com/x/onerror=alert(1)//](https://n.invalid/x.png)',
  '![a](https://n.invalid/x.png "看https://x.com/x/style=position:fixed//")',
  '[看https://evil.com](https://a.com)',
  // Raw HTML the model wrote: the autolink output must be sanitized together
  // with the tag it landed in, not spliced back in afterwards.
  '<img src="https://n.invalid/x.png" alt="x https://x.com/x/onerror=alert(1)//">',
  '<img src="https://n.invalid/x.png" alt="看https://x.com/x/onerror=alert(1)//">',
  '<a href="https://a.com" title="x https://x.com/x/onmouseover=alert(1)//">t</a>',
  '<details><summary>s</summary>[看https://x.com/x/onerror=alert(1)//](https://a.com) ![x https://x.com/x/onerror=alert(1)//](https://n.invalid/x.png)</details>',
  // Positive control: proves the sanitizer ran at all in this harness.
  '<img src=x onerror=alert(1)> https://a.com',
];

(async () => {
  const origin = process.env.ODYSSEUS_TEST_STATIC_ORIGIN;
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.route('**/autolink-harness', route => route.fulfill({ contentType: 'text/html', body: '<!doctype html><div id="host"></div>' }));
    await page.route('**/api/**', route => route.fulfill({ json: { fonts: {}, value: null } }));
    await page.goto(origin + '/autolink-harness');
    const flagged = await page.evaluate(async (cases) => {
      const markdown = await import('/static/js/markdown.js');
      const host = document.getElementById('host');
      const flagged = [];
      for (const source of cases) {
        host.innerHTML = markdown.mdToHtml(source);
        const bad = [];
        for (const el of host.querySelectorAll('*')) {
          for (const attr of el.attributes) {
            const name = attr.name.toLowerCase();
            if (name.startsWith('on') || name === 'style' || !/^[a-z][a-z0-9-]*$/.test(name)) bad.push(`${el.tagName}.${attr.name}`);
          }
        }
        const nested = host.querySelectorAll('a a').length;
        if (bad.length || nested) flagged.push({ source, bad, nested });
      }
      return flagged;
    }, cases);
    console.log(JSON.stringify({ flagged }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
