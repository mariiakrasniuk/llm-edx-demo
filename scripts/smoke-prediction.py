"""Experiment 3 works directly, and errors remain visible in that experiment."""
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1280,'height':1000})
    page.set_default_timeout(90000)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8000/#predict')
    assert page.locator('#q-sample').is_disabled()
    start=page.locator('#q-prediction-form button')
    start.click()
    page.wait_for_function("document.querySelector('#q-prediction-status').textContent.includes('Real forward pass')")
    assert page.locator('#q-current-text').inner_text()=='Text so far: The sky is'
    initial=page.locator('#q-tokens .token').count()
    for i in range(2):
        page.locator('#q-sample').click()
        page.wait_for_function("!document.querySelector('#q-sample').disabled")
        assert page.locator('#q-tokens .token').count()==initial+i+1
    print('Two sampled tokens appended:',page.locator('#q-current-text').inner_text(),flush=True)
    previous=page.locator('#q-current-text').inner_text()
    page.route('**/inspect',lambda route:route.fulfill(status=503,content_type='application/json',body='{"detail":"Test model unavailable"}'))
    start.click()
    page.wait_for_function("document.querySelector('#q-prediction-status').textContent.includes('Test model unavailable')")
    assert start.is_enabled()
    assert page.locator('#q-current-text').inner_text()==previous
    page.unroute('**/inspect')
    start.click()
    page.wait_for_function("document.querySelector('#q-prediction-status').textContent.includes('Real forward pass')")
    page.locator('#predict').screenshot(path='artifacts/prediction-direct.png')
    assert not errors,errors
    browser.close()
    print('Direct start, sampling, visible error, preserved results and retry passed.')
