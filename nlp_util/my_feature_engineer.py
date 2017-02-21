# -*- coding: utf-8 -*-
# @Time    : 17/2/17 下午5:50
# @Author  : liulei
# @Brief   : 
# @File    : my_feature_engineer.py
# @Software: PyCharm Community Edition
import re
import jieba

##替换常用HTML字符实体.
# 使用正常的字符替换HTML中特殊的字符实体.
# 你可以添加新的实体字符到CHAR_ENTITIES中,处理更多HTML字符实体.
# @param htmlstr HTML字符串.
def replaceCharEntity(htmlstr):
    CHAR_ENTITIES = {'nbsp': ' ', '160': ' ',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"', '34': '"',}

    re_charEntity = re.compile(r'&#?(?P<name>\w+);')
    sz = re_charEntity.search(htmlstr)
    while sz:
        entity = sz.group()  # entity全称，如&gt;
        key = sz.group('name')  # 去除&;后entity,如&gt;为gt
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            # 以空串代替
            htmlstr = re_charEntity.sub('', htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
    return htmlstr


##过滤HTML中的标签
# 将HTML中标签等信息去掉
# @param htmlstr HTML字符串.
def filter_tags(htmlstr):
    # 先过滤CDATA
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)  # 匹配CDATA
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)  # Script
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)  # style
    re_br = re.compile('<br\s*?/?>')  # 处理换行
    re_h = re.compile('</?\w+[^>]*>')  # HTML标签
    re_comment = re.compile('<!--[^>]*-->')  # HTML注释
    s = re_cdata.sub('', htmlstr)  # 去掉CDATA
    s = re_script.sub('', s)  # 去掉SCRIPT
    s = re_style.sub('', s)  # 去掉style
    s = re_br.sub('\n', s)  # 将br转换为换行
    s = re_h.sub('', s)  # 去掉HTML 标签
    s = re_comment.sub('', s)  # 去掉HTML注释
    # 去掉多余的空行
    blank_line = re.compile('\n+')
    s = blank_line.sub('\n', s)
    s = replaceCharEntity(s)  # 替换实体
    return s

def remove_stopwords(str):
    words = jieba.cut(str)
    stopwords = {}.fromkeys(line.rstrip() for line in open('stopwords.txt'))
    sw_set = set(stopwords)
    final_words = []
    for w in words:
        if not w.encode('utf-8') in sw_set:
            final_words.append(w)
    return ''.join(final_words)

def filterPOS(org_text):
    txtlist = []
    POS = ['zg', 'uj', 'ul', 'e', 'd', 'uz', 'y']
    # 去除特定词性的词
    for w in org_text:
        if w.flag in POS:
            pass
        else:
            txtlist.append(w.word)
    return txtlist


#过滤url
def filter_url(str):
    if str.find('www') < 0 and str.find('http') < 0:
        return str
    myString_list = [item for item in str.split(" ")]
    url_list = []
    for item in myString_list:
        try:
            url_list.append(re.search("(?P<url>https?://[^\s]+)", item).group("url"))
        except:
            pass
    for i in url_list:
        str = str.replace(i, ' ')
    return str


# 检查某字符是否分句标志符号的函数；如果是，返回True，否则返回False
def FindToken(cutlist, char):
    if char in cutlist:
        return True
    else:
        return False

        # 进行分句的核心函数


# 设置分句的标志符号；可以根据实际需要进行修改
#cutlist = "。！？".decode('utf-8')
#cutlist = "。！？.!?。!?".decode('utf-8')
cutlist = "。！？!?。!?".decode('utf-8') #不包含英文句号,因为也会被当成小数点
#字符串分句
def Cut(lines, cutlist=cutlist):  # 参数1：引用分句标志符；参数2：被分句的文本，为一行中文字符
    l = []  # 句子列表，用于存储单个分句成功后的整句内容，为函数的返回值
    line = []  # 临时列表，用于存储捕获到分句标志符之前的每个字符，一旦发现分句符号后，就会将其内容全部赋给l，然后就会被清空


    if lines.find('http') > 0:
        myString_list = [item for item in lines.split(" ")]
        for item in myString_list:
            try:
                url = re.search("(?P<url>https?://[^\s]+)", item).group("url")
                l.append(url)
                lines = lines.replace(url, ' ')
            except:
                pass

    for i in lines:  # 对函数参数2中的每一字符逐个进行检查 （本函数中，如果将if和else对换一下位置，会更好懂）
        if FindToken(cutlist, i):  # 如果当前字符是分句符号
            line.append(i)  # 将此字符放入临时列表中
            l.append(''.join(line))  # 并把当前临时列表的内容加入到句子列表中
            line = []  # 将符号列表清空，以便下次分句使用
        else:  # 如果当前字符不是分句符号，则将该字符直接放入临时列表中
            line.append(i)
    if len(line) != 0:
        l.append(''.join(line))
    return l
