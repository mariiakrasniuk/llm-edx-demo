"""Live browser smoke test. Run against an already-started trained Qwen server."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto('http://127.0.0.1:8000')
    page.get_by_role('button', name='Inspect the model').click()
    page.wait_for_function("document.querySelector('#status').textContent.includes('Real forward pass')", timeout=180000)
    initial = page.locator('#tokens .token').count()
    assert initial > 0
    page.locator('#layer').select_option('10')
    page.wait_for_function("!document.querySelector('#layer').disabled")
    page.locator('#head').select_option('3')
    page.wait_for_function("!document.querySelector('#head').disabled")
    assert 'Layer 11, head 4' in page.locator('#selection').inner_text()
    page.locator('#greedy').click()
    page.wait_for_function("!document.querySelector('#greedy').disabled")
    assert page.locator('#tokens .token').count() == initial + 1
    page.locator('#undo').click()
    assert page.locator('#tokens .token').count() == initial
    page.locator('#chat-form button').click()
    page.wait_for_function("document.querySelector('#reply').textContent.length > 0", timeout=120000)
    reply = page.locator('#reply').inner_text()
    Path('artifacts').mkdir(exist_ok=True)
    page.screenshot(path='artifacts/qwen-desktop.png', full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path='artifacts/qwen-mobile.png', full_page=True)
    assert not errors, errors
    print(json.dumps({'tokens': initial, 'reply': reply, 'browser_errors': errors, 'mobile_overflow': False}))
    browser.close()
