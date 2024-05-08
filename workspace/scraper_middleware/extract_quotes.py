

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from playwright_stealth import stealth_sync

def run(playwright, author_url):
    server = '50.118.217.215:60000'
    username = "upstreamcommerce2"
    password = "4Y6zkSMT8V"
    proxy_options = {
        'server' : server,
        'username' : username,
        'password' : password,
    }

    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0_1; Valve Steam GameOverlay/1679680416) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
    
    browser = playwright.firefox.launch(headless=True)  # Set headless=False to see the browser
    context = browser.new_context(proxy = proxy_options, user_agent = user_agent)
    page = context.new_page()
    stealth_sync(page)
    page.goto(author_url)
    if page.url != author_url:
        return None

    page.wait_for_load_state()
    page.wait_for_timeout(3000)

    html = page.inner_html('body')
    soup = BeautifulSoup(html, 'html.parser')

    result = []

    for itr in range(10):

        eles = soup.select_one('div#qbc{}'.format(str(itr)))
        if eles is None:
            continue
        eles = eles.select('div.grid-item.qb.clearfix.bqQt')

        for ele in eles:

            q1 = ele.select_one('div')
            q2 = q1.text.replace('\n','').strip()
            if q2 is None or q2 == '':
                continue
            result.append(q2)


    browser.close()

    return list(set(result))

def extract_quotes(author_url):

    quotes = []
    with sync_playwright() as playwright:
        itr = 1
        while True:
            ret = None
            if itr == 1:
                ret = run(playwright, author_url)
            else:
                ret = run(playwright, author_url + '_{}'.format(str(itr)))
            if ret is None:
                break
            quotes.extend(ret)
            itr += 1

    return list(set(quotes))

