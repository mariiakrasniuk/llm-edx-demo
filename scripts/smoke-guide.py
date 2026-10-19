"""Verify the shorter default path and progressive disclosure in a real browser."""
from playwright.sync_api import sync_playwright
import json
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000})
    page.set_default_timeout(120000)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8000')
    assert not page.locator('#chat-form').is_visible()
    assert not page.locator('#q-delta').is_visible()
    assert page.locator('.finish-step').count()==4
    assert page.evaluate('new Set([...document.querySelectorAll("[id]")].map(n=>n.id)).size === document.querySelectorAll("[id]").length')
    words=len(page.locator('main').inner_text().split())
    page.screenshot(path='artifacts/guided-lab-desktop.png')
    page.locator('#q-inspect-form button[type=submit]').click()
    page.wait_for_function("document.querySelector('#q-status').textContent.includes('Real forward pass')")
    assert page.locator('#q-tokens .token').count()>0
    page.locator('[data-q-example="The river bank"]').click()
    page.wait_for_function("!document.querySelector('#q-prompt').disabled")
    page.locator('#q-tokens .token').last.click()
    page.locator('#attention').screenshot(path='artifacts/guided-lab-attention.png')
    page.locator('#q-greedy').click()
    page.wait_for_function("!document.querySelector('#q-greedy').disabled")
    assert 'Text so far:' in page.locator('#q-current-text').inner_text()
    page.locator('#compare-contexts').click()
    page.wait_for_function("document.querySelector('#compare-status').textContent.includes('Complete.')")
    for button in page.locator('.finish-step').all():button.click()
    assert 'All four' in page.locator('#journey-progress').inner_text()
    page.locator('[data-check=training]').click()
    assert 'Not quite' in page.locator('#check-feedback').inner_text()
    page.locator('[data-check=context]').click()
    assert 'Exactly' in page.locator('#check-feedback').inner_text()
    page.get_by_text('Follow a transformer block and change one weight',exact=True).click()
    assert page.locator('#q-delta').is_visible()
    page.set_viewport_size({'width':390,'height':844})
    page.evaluate('window.scrollTo(0,0)')
    page.screenshot(path='artifacts/guided-lab-mobile.png')
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors,errors
    print(json.dumps({'core_path':'passed','visible_words_before_inference':words,'optional_tools':'accessible','mobile_overflow':False,'js_errors':errors}))
    browser.close()
