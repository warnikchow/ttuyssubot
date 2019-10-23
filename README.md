# Ttuyssubot
Contextual Spacing for Conversation-style (and non-normalized) Text

## Requirements
fasttext, Keras (TensorFlow), Numpy

## Word Vector 
[Pretrained 100dim fastText vector](https://drive.google.com/open?id=1jHbjOcnaLourFzNuP47yGQVhBTq6Wgor)
* Download this and unzip THE FOLDER in the same folder with 'csct.py' 
* Loading the model will be processed by load_model('vectors/model')

## System Description
* Easy start: Python3 execute file
<pre><code> python3 csct.py </code></pre>
* This system assigns a contextual spacing for conversation-style and non-normalized Korean text
- ex1) 아버지친구분당선되셨더라 >> "아버지 친구분 당선 되셨더라"
- ex2) 너본지꽤된듯 >> "너 본지 꽤 된 듯"
- ex3) 뭣이중헌지도모름서 >> "뭣이 중헌지도 모름서"
- ex4) 나얼만큼사랑해 >> "나 얼만큼 사랑해"
* The spacing may not be strictly correct, but the system was trained in a way to give a plausible duration for speech synthesis, in the aspect of a non-canonical spoken language.
* Importing automatic spacer
<pre><code> from csct_dist import correct as cor </code></pre>

## Reference (as a toolkit)
* 조원익, 천성준, 김지원, 김남수, "문장 정보를 고려한 딥 러닝 기반 자동 띄어쓰기의 개념 및 활용," 제30회 한글 및 한국어 정보처리 학술대회 논문집, 2018, pp. 181-184. [[Paper]](http://www.koreascience.or.kr/article/CFKO201832073078638.page) [[Slide]](https://www.slideshare.net/WonIkCho/warnik-chow-2018-hclt-119690256)
```
@inproceedings{cho2018concept,
  title={Concept and Application of Deep learning-based Automatic Spacing},
  author={Cho, Won Ik and Cheon, Sung Jun and Kim, Ji Won and Kim, Nam Soo},
  booktitle={Annual Conference on Human and Language Technology},
  pages={181--184},
  year={2018},
  organization={Human and Language Technology}
}
```
* For English version, check [RAWS](https://github.com/warnikchow/raws)
### DISCLAIMER: This model is trained with drama scripts and targets user-generated noisy texts; for the accurate spacing of literary style texts, refer to [PyKoSpacing](https://github.com/haven-jeon/PyKoSpacing)

## Demonstration
* https://www.youtube.com/watch?v=mcPZVpKCH94&feature=youtu.be
