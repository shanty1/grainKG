import os,json,time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

settings = {
    "recentDestinations": [{
        "id": "Save as PDF",
        "origin": "local",
        "account": ""
    }],
    "selectedDestinationId": "Save as PDF",
    "version": 2,
    "isHeaderFooterEnabled": False,

    # "customMargins": {},
    "marginsType": 1,
    "scalingType": 3,   # 0默认 1适合可打印区域 2适合纸张大小 3自定义
    "scaling": 100,
    "scalingTypePdf": 1,
    "isLandscapeEnabled":True,#landscape横向，portrait 纵向，若不设置该参数，默认纵向
    "isCssBackgroundEnabled": True,
    "mediaSize": {
        "height_microns": 594000,
        "name": "ISO_A2",
        "width_microns": 420000,
        "custom_display_name": "A2",
        "is_default": True,
        "vendor_id":"66"
    },
}
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--enable-print-browser')
# chrome_options.add_argument('--headless') #headless模式下，浏览器窗口不可见，可提高效率
chrome_options.add_argument('--kiosk-printing')  # 静默打印，无需用户点击打印页面的确定按钮
chromedriver_file = f'{os.path.abspath(os.path.dirname(__file__))}\\chromedriver.exe'

def print_web_to_file(url, save_dir, filename):
    # 设置参数，保存路径
    prefs = {
        'printing.print_preview_sticky_settings.appState': json.dumps(settings),
        'savefile.default_directory': save_dir #此处填写你希望文件保存的路径
    }
    chrome_options.add_experimental_option('prefs', prefs)
    # 打开网页
    driver = webdriver.Chrome(chromedriver_file,options=chrome_options)
    driver.maximize_window()
    driver.implicitly_wait(10)	#隐式等待10s查询元素
    driver.get(url)
    time.sleep(1)
    driver.execute_script('document.title="'+filename+'";window.print();') #利用js修改网页的title，该title最终就是PDF文件名，利用js的window.print可以快速调出浏览器打印窗口，避免使用热键ctrl+P
    driver.close()

if __name__ == '__main__':
    url = "https://echarts.apache.org/examples/zh/index.html"
    save_dir = "/resources/report_dir"
    fielname = "report"
    print_web_to_file(url, save_dir, fielname)
