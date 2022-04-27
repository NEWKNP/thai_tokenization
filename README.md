# THAI tokenization
collecting THAI tokenization in nlp task from all resourses.

## จุดประสงค์
1. รวบรวม tokenization สำหรับภาษาไทย ที่ใช้งานได้ในปัจจุบัน (04/2022)
2. ทั้งนี้ทั้งนั้น ไม่มีตัวไหนดีที่สุด แต่มีตัวที่เหมาะสมกับงานของเรามากที่สุด เช่น ถ้าการ deploy อิงความเร็วเป็นหลัก

## Tokenization
Tokenization หรือการตัดคำ คือการรับ text ประโยค บทความ หรือเอกสารต่างๆ มาซอยออกมาเป็นคำๆ ซึ่งเราสามารถนำกลุ่มคำเหล่านี้มาใช้ในงานต่างๆ เช่น การวิเคราะห์ข้อมูล การทำ word cloud หรือนำไปใช้ใน AI อย่าง chatbot ที่ทำงานโดย machine learning ที่รับได้แค่ตัวเลขเท่านั้น เราจึงนำกลุ่มคำเข้า word embedding  
การตัดคำแบ่งออกเป็น level ตั้งแต่ระดับ ประโยค(sentence) ไปจนถึง พยางค์(syllabel)  
(ส่วน phonetic สำหรับงานเสียง จะอัพเดทในอนาคต)  

การติดตั้ง library เบื้องต้น
```.bash
pip install -q python-crfsuite pythainlp[full]
pip install -q transformers[sentencepiece]
pip install deepcut
```

### sentence level
การตัดเหลือเป็น ประโยค จากบทความ ข่าว กระทู้  
```.python
from pythainlp.tokenize import sent_tokenize
text = "อะจ๊ะเอ๋ตัวเอง ท่านผู้เจริญผู้ซึ่งมากไปด้วยปัญญา อะฮ่า"
```
* CRFCut - Thai sentence segmenter  
Thai sentence segmentation using conditional random field, default model trained on TED dataset  
Ref: https://github.com/vistec-AI/ted_crawler
```.python
print(sent_tokenize(text)) 
#-> ['อะจ๊ะเอ๋ตัวเอง ', 'ท่านผู้เจริญผู้ซึ่งมากไปด้วยปัญญา ', 'อะฮ่า']
```
* whitespace - split by whitespaces. Specifiaclly, with `regex` pattern `r" +"`
```.python
print(sent_tokenize(text, engine="whitespace"))
#-> ['อะจ๊ะเอ๋ตัวเอง', 'ท่านผู้เจริญผู้ซึ่งมากไปด้วยปัญญา', 'อะฮ่า']
```
* whitespace+newline - split by whitespaces and newline.
```.python
# whitespace+newline engine
print(sent_tokenize(text, engine="whitespace+newline"))
#-> ['อะจ๊ะเอ๋ตัวเอง', 'ท่านผู้เจริญผู้ซึ่งมากไปด้วยปัญญา', 'อะฮ่า']
```
* tltk - split by TLTK
```.python
print(sent_tokenize(text, engine="tltk"))
#-> ['อะจ๊ะเอ๋ตัวเอง ท่านผู้เจริญผู้ซึ่งมากไปด้วยปัญญา อะฮ่า']
```
### clause level
Clause tokenizer. (or Clause segmentation)  
Tokenizes running word list into list of clauses (list of strings). split by CRF trained on LST20 Corpus  
*input ที่ควรเข้าต้องผ่าน word tokenization มาก่อน  
```.python
from pythainlp.tokenize import clause_tokenize
print(clause_tokenize(['อะ','จ๊ะเอ๋','ตัวเอง','ท่าน','ผู้เจริญ','ผู้','ซึ่ง','มาก','ไป','ด้วย','ปัญญา','อะฮ่า']))
#-> [['อะ', 'จ๊ะเอ๋', 'ตัวเอง', 'ท่าน', 'ผู้เจริญ', 'ผู้'], ['ซึ่ง', 'มาก', 'ไป', 'ด้วย', 'ปัญญา', 'อะฮ่า']]
```
### word level
#### dictionary base
* longest maching
ตัดคำให้ยาวที่สุด
```.python
print('|'.join(word_tokenize('ถอยหมอน', engine='longest')))
#-> ถอย|หมอน
print('|'.join(word_tokenize(text, engine='longest')))
#-> อะ|จ๊ะเอ๋|ตัวเอง| |ท่าน|ผู้|เจริญ|ผู้|ซึ่ง|มาก|ไป|ด้วย|ปัญญา| |อะ|ฮ่า
```
* newmm  
maximal maching + Thai Character Cluster
```.python
print('|'.join(word_tokenize('ถอยหมอน', engine='newmm')))
#-> ถอย|หมอน
print('|'.join(word_tokenize(text, engine='newmm')))
#-> อะ|จ๊ะเอ๋|ตัวเอง| |ท่าน|ผู้|เจริญ|ผู้|ซึ่ง|มาก|ไป|ด้วย|ปัญญา| |อะ|ฮ่า
```
* nercut  
maximal maching + Thai Character Cluster (TCC) boundaries
```.python
print('|'.join(word_tokenize('ถอยหมอน', engine='nercut')))
#-> ถอย|หมอน
print('|'.join(word_tokenize(text, engine='nercut')))
#-> อะ|จ๊ะเอ๋|ตัวเอง| |ท่าน|ผู้|เจริญ|ผู้|ซึ่ง|มาก|ไป|ด้วย|ปัญญา| |อะ|ฮ่า
```
* nlpo3  
maximal maching + Thai Character Cluster boundaries for rust/node.js
```.python
print('|'.join(word_tokenize('ถอยหมอน', engine='nlpo3')))
#-> ถอย|หมอน
print('|'.join(word_tokenize(text, engine='nlpo3')))
#-> อะ|จ๊ะเอ๋|ตัวเอง| |ท่าน|ผู้|เจริญ|ผู้|ซึ่ง|มาก|ไป|ด้วย|ปัญญา| |อะ|ฮ่า
```
* multi_cut
```.python
print('|'.join(word_tokenize('ถอยหมอน', engine='multi_cut')))
#-> ถอย|หมอน
print('|'.join(word_tokenize(text, engine='multi_cut')))
#-> อะ|จ๊ะเอ๋|ตัวเอง| |ท่าน|ผู้|เจริญ|ผู้|ซึ่ง|มาก|ไป|ด้วย|ปัญญา| |อะ|ฮ่า
```
#### machine learning base
* sefr cut  
Stacked Ensemble Filter and Refine for Word Segmentation
```.python
print('|'.join(word_tokenize('ถอยหมอน', engine='sefr_cut')))
#-> ถอย|หมอน
print('|'.join(word_tokenize(text, engine='sefr_cut')))
#-> อะ|จ๊ะ|เอ๋|ตัว|เอง| |ท่าน|ผู้|เจริญ|ผู้|ซึ่ง|มาก|ไป|ด้วย|ปัญญา| |อะ|ฮ่า
```
#### deep learning base
* Attacut
```.python
print('|'.join(word_tokenize('ถอยหมอน', engine='attacut')))
#-> ถอย|หมอน
print('|'.join(word_tokenize(text, engine='attacut')))
#-> อะจ๊ะเอ๋|ตัว|เอง| |ท่าน|ผู้|เจริญ|ผู้|ซึ่ง|มาก|ไป|ด้วย|ปัญญา| |อะฮ่า
```
* Deepcut
```.python
print('|'.join(word_tokenize('ถอยหมอน', engine='deepcut')))
#-> ถอยหมอน
print('|'.join(word_tokenize(text, engine='deepcut')))
#-> อะจ๊ะ|เอ๋|ตัว|เอง| |ท่าน|ผู้|เจริญ|ผู้|ซึ่ง|มาก|ไป|ด้วย|ปัญญา อะฮ่า
```
* OSKut
```.python
print('|'.join(word_tokenize('ถอยหมอน', engine='oskut')))
#-> ถอยหมอน
print('|'.join(word_tokenize(text, engine='oskut')))
#-> อะ|จ๊ะ|เอ๋|ตัว|เอง| |ท่าน|ผู้|เจริญ|ผู้ซึ่ง|มาก|ไป|ด้วย|ปัญญา| |อะ|ฮ่า
```
### subword level
* tcc
```.python
print('|'.join(subword_tokenize('ถอยหมอน', engine='tcc')))
#-> ถ|อ|ย|ห|ม|อ|น
print('|'.join(subword_tokenize(text, engine='tcc')))
#-> อะ|จ๊ะ|เอ๋|ตัว|เอ|ง| |ท่า|น|ผู้|เจ|ริ|ญ|ผู้|ซึ่|ง|มา|ก|ไป|ด้|ว|ย|ปัญ|ญา| |อะ|ฮ่า
```
* etcc
```.python
print('|'.join(subword_tokenize('ถอยหมอน', engine='etcc')))
#-> ถ|อ|ย|ห|ม|อ|น
print('|'.join(subword_tokenize(text, engine='etcc')))
#-> อะ|จ๊ะ|เอ๋ตัวเอ|ง| |ท่า|น|ผู้เจ|ริญ|ผู้|ซึ่|ง|มา|ก|ไป|ด้|ว|ย|ปัญ|ญา| |อะ|ฮ่า
```
* dict
```.python
print('|'.join(subword_tokenize('ถอยหมอน', engine='dict')))
#-> ถอย|หมอน
print('|'.join(subword_tokenize(text, engine='dict')))
#-> อะ|จ๊ะ|เอ๋|ตัว|เอง| |ท่าน|ผู้|เจริญ|ผู้|ซึ่ง|มาก|ไป|ด้วย|ปัญ|ญา| |อะ|ฮ่า
```
* ssg
```.python
print('|'.join(subword_tokenize('ถอยหมอน', engine='ssg')))
#-> ถอย|หมอน
print('|'.join(subword_tokenize(text, engine='ssg')))
#-> อะ|จ๊ะ|เอ๋|ตัว|เอง| ท่าน|ผู้|เจริญ|ผู้|ซึ่ง|มาก|ไป|ด้วย|ปัญ|ญา| |อะ|ฮ่า
```
* tltk
```.python
print('|'.join(subword_tokenize('ถอยหมอน', engine='tltk')))
#-> ถอย|หมอน
print('|'.join(subword_tokenize(text, engine='tltk')))
#-> อะ|จ๊ะ|เอ๋|ตัว|เอง|<s/>ท่าน|ผู้|เจริญ|ผู้|ซึ่ง|มาก|ไป|ด้วย|ปัญ|ญา|<s/>อะ|ฮ่า
```
* wangchanberta
```.python
print('|'.join(subword_tokenize('ถอยหมอน', engine='wangchanberta')))
#-> ▁|ถอย|หมอน
print('|'.join(subword_tokenize(text, engine='wangchanberta')))
#-> ▁|อะ|จ๊ะ|เอ๋|ตัวเอง|▁ท่าน|ผู้|เจริญ|ผู้ซึ่ง|มาก|ไปด้วย|ปัญญา|▁|อะ|ฮ่า
```
### syllabel level
เป็นวิธีการทำ token ที่ง่ายที่สุด เพราะ string มีโครงสร้างคล้าย array of characters จึงใช้ list ครอบ string แตกออกมา character ได้ใน code 1 บรรทัด
```.python
print(list(text))
#-> ['อ', 'ะ', 'จ', '๊', 'ะ', 'เ', 'อ', '๋', 'ต', 'ั', 'ว', 'เ', 'อ', 'ง', ' ', 'ท', '่', 'า', 'น', 'ผ', 'ู', '้', 'เ', 'จ', 'ร', 'ิ', 'ญ', 'ผ', 'ู', '้', 'ซ', 'ึ', '่', 'ง', 'ม', 'า', 'ก', 'ไ', 'ป', 'ด', '้', 'ว', 'ย', 'ป', 'ั', 'ญ', 'ญ', 'า', ' ', 'อ', 'ะ', 'ฮ', '่', 'า']
```
ในการทำ syllabel token จะไม่สนโครงสร้างทางภาษา แต่จะสนความน่าจะเป็นร่วมของแต่ละ character เช่น อักษร-สระ ซึ่งจะช่วยแก้ปัญหากับ การสะกดผิด และ คำหายาก แต่ข้อจำกัดคือ linguistic structures such as words need to be learned from the data. This requires significant compute, memory, and data ทำให้ syllabel token ไม่เป็นที่นิยมเท่า word token หรือ subword token
  
syllable_tokenize in pythainlp now is deprecated (from developer)  

# Reference resourse
1. pythainlp: https://pythainlp.github.io/dev-docs/api/tokenize.html
2. nlp with transformer: https://github.com/nlp-with-transformers/notebooks
