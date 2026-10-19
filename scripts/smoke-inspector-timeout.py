"""Verify stalled-request recovery without waiting 30 real seconds."""
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page()
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8000')
    submit=page.locator('#q-inspect-form button[type=submit]')
    submit.click()
    page.wait_for_function("document.querySelector('#q-status').textContent.includes('Real forward pass')")
    previous=page.locator('#q-tokens').inner_text()
    page.clock.install()
    page.route('**/inspect',lambda route:None)
    submit.click()
    assert page.locator('#q-cancel').is_enabled()
    page.clock.fast_forward(31000)
    page.wait_for_function("document.querySelector('#q-status').textContent.includes('30 seconds')")
    assert submit.is_enabled()
    assert page.locator('#q-tokens').inner_text()==previous
    submit.click()
    page.locator('#q-cancel').click()
    page.wait_for_function("document.querySelector('#q-status').textContent.includes('cancelled')")
    assert submit.is_enabled()
    assert page.locator('#q-tokens').inner_text()==previous
    page.unroute('**/inspect')
    submit.click()
    page.wait_for_function("document.querySelector('#q-status').textContent.includes('Real forward pass')")
    assert not errors,errors
    print('Live inference, simulated timeout, cancellation, preserved results and successful retry passed.')
    browser.close()
