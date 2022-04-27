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
CRFCut - Thai sentence segmenter  
Thai sentence segmentation using conditional random field, default model trained on TED dataset  
Ref: https://github.com/vistec-AI/ted_crawler
```.python
print(sent_tokenize(text)) 
#-> ['อะจ๊ะเอ๋ตัวเอง ', 'ท่านผู้เจริญผู้ซึ่งมากไปด้วยปัญญา ', 'อะฮ่า']
```
whitespace - split by whitespaces. Specifiaclly, with `regex` pattern `r" +"`
```.python
print(sent_tokenize(text, engine="whitespace"))
#-> ['อะจ๊ะเอ๋ตัวเอง', 'ท่านผู้เจริญผู้ซึ่งมากไปด้วยปัญญา', 'อะฮ่า']
```
whitespace+newline - split by whitespaces and newline.
```.python
# whitespace+newline engine
print(sent_tokenize(text, engine="whitespace+newline"))
#-> ['อะจ๊ะเอ๋ตัวเอง', 'ท่านผู้เจริญผู้ซึ่งมากไปด้วยปัญญา', 'อะฮ่า']
```
tltk - split by TLTK
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
#### machine learning base
#### deep learning base
### subword level

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
