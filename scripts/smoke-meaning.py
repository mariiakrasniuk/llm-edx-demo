"""Check the beginner meaning comparison on the real local model."""
import json
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100})
    page.set_default_timeout(120000)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8000')
    assert not page.locator('#q-head-view').is_visible()
    page.locator('#meaning-form button[type=submit]').click()
    page.wait_for_function("document.querySelector('#meaning-status').textContent.startsWith('Both answers')")
    answers=page.locator('.meaning-answer>p:not(.fine)').all_text_contents()
    assert len(answers)==2 and all(answers)
    page.locator('#attention').screenshot(path='artifacts/meaning-comparison.png')
    page.locator('#meaning-control').click()
    assert page.locator('#meaning-results').inner_text()==''
    page.locator('#meaning-form button[type=submit]').click()
    page.wait_for_function("document.querySelector('#meaning-status').textContent.startsWith('Both answers')")
    control=page.locator('.meaning-answer>p:not(.fine)').all_text_contents()
    assert control[0]==control[1],control
    assert 'identical' in page.locator('#meaning-observation').inner_text()
    page.get_by_text('Optional: see the attention connections (Head View)',exact=True).click()
    assert page.locator('#q-layer').is_visible()
    page.locator('#q-inspect-form button[type=submit]').click()
    page.wait_for_function("document.querySelector('#q-status').textContent.includes('Real forward pass')")
    assert page.locator('#q-head-view svg').is_visible()
    assert 'self-connection' in page.locator('.attention-reading').inner_text()
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    assert not errors,errors
    print(json.dumps({'answers':answers,'identical_control':'passed','optional_head_view':'passed','js_errors':errors}))
    browser.close()
