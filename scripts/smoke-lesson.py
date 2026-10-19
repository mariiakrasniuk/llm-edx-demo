"""Exercise the complete lesson against the real running Qwen checkpoint."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    page.set_default_timeout(180000)
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto('http://127.0.0.1:8000')
    assert page.locator('#chat').count() == 1
    page.evaluate("document.querySelectorAll('details').forEach(d=>d.open=true)")
    page.locator('#q-inspect-form button[type=submit]').click()
    page.wait_for_function("document.querySelector('#q-status').textContent.includes('Real forward pass')", timeout=180000)
    print('Inspector loaded', flush=True)
    initial = page.locator('#q-tokens .token').count()
    page.locator('#q-layer').select_option('10')
    page.wait_for_function("!document.querySelector('#q-layer').disabled")
    page.locator('#q-all-heads').click()
    assert page.locator('#q-head-controls [aria-pressed=true]').count() == 32
    page.locator('#q-one-head').click()
    page.locator('#q-tokens .token').last.click()
    page.locator('#q-greedy').click()
    page.wait_for_function("!document.querySelector('#q-greedy').disabled")
    assert page.locator('#q-tokens .token').count() == initial+1
    page.locator('#q-undo').click()
    assert page.locator('#q-tokens .token').count() == initial
    page.locator('#q-delta').fill('1')
    page.locator('#q-apply-weight').click()
    page.wait_for_function("document.querySelector('#q-weight-result').textContent.includes('restored')")
    page.locator('#q-reset-weight').click()
    page.wait_for_function("!document.querySelector('#q-reset-weight').disabled")
    page.locator('#q-window-run').click()
    page.wait_for_function("document.querySelector('#q-window-status').textContent.includes('Compared')")
    print('Weights, heads, append, undo and context window passed', flush=True)
    page.locator('#prepare').click()
    page.wait_for_function("document.querySelector('#run-status').textContent.includes('Ready:')", timeout=180000)
    page.locator('#step').click()
    assert '1 /' in page.locator('#step-count').inner_text()
    page.locator('#undo').click()
    assert '0 /' in page.locator('#step-count').inner_text()
    print('Generation playback passed', flush=True)
    page.locator('#compare-contexts').click()
    page.wait_for_function("document.querySelector('#compare-status').textContent.includes('Complete.')", timeout=180000)
    page.locator('#style-form button[type=submit]').click()
    page.wait_for_function("document.querySelector('#style-status').textContent.includes('4 of 4')", timeout=180000)
    print('Context comparison and all styles passed', flush=True)
    page.locator('#chat-message').fill('Describe a sunny morning in one sentence.')
    page.locator('#chat-form button[type=submit]').click()
    page.wait_for_function("document.querySelector('#chat-status').textContent.includes('4 of 4')", timeout=180000)
    page.get_by_role('button', name='Continue with this reply').first.click()
    assert page.locator('#chat-history .history-turn').count() == 2
    print('Four chat replies and shared history passed', flush=True)
    Path('artifacts').mkdir(exist_ok=True)
    page.locator('#attention').screenshot(path='artifacts/full-lesson-head-view.png')
    page.screenshot(path='artifacts/full-lesson-desktop.png')
    page.set_viewport_size({'width': 390, 'height': 844})
    page.locator('#attention').scroll_into_view_if_needed()
    page.screenshot(path='artifacts/full-lesson-mobile.png')
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'mobile overflow'
    assert not errors, errors
    print(json.dumps({'initial_tokens': initial, 'browser_errors': errors, 'all_activities': 'passed'}))
    browser.close()
