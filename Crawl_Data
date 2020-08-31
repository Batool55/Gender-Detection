import requests
from bs4 import BeautifulSoup


#This piece of code used to download all .wav files from festvox website and save each database in seperate folder
urls = ['http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_bdl_arctic/wav/',
        'http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_slt_arctic/wav/',
        'http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_clb_arctic/wav/',
        'http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_rms_arctic/wav/']

for url in urls:
    fn = url[48:51]
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, "lxml")
    for link in soup.find_all("a"):
        i = link.get("href")
        if (".wav") in i:
            s = url+i
            w = requests.get(s,allow_redirects=True)
            path = fn + '/' + i
            open(path, 'wb').write(w.content)
     
