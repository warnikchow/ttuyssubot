# CSCT
Contextual Spacing for Conversation-style (and non-normalized) Text

## Requirements
fasttext, Keras (TensorFlow), Numpy

## Word Vector 
https://drive.google.com/file/d/1PE7RWIjLyBaBdrLg_ybUpnoJAvibkrXY/view?usp=sharing
* Download this and unzip THE FOLDER in the same folder with 'csct.py' 
* Loading the model will be processed by load_model('vectors/model')

## System Description
* Easy start: Python3 execute file
<pre><code> python3 csct.py </code></pre>
* This system assigns a contextual spacing for conversation-style and non-normalized Korean text
- ex1) 아버지친구분당선되셨더라 >> "아버지 친구분 당선되셨더라"
- ex2) 너본지꽤된듯 >> "너 본지 꽤 된듯"
- ex3) 뭣이중헌지도모름서 >> "뭣이 중헌지도 모름서"
- ex4) 엄마가죽을병에넣어뒀어 >> "엄마가 죽을 병에 넣어뒀어"
- ex5) 나얼만큼사랑해 >> "나 얼만큼 사랑해"
* The spacing may not be strictly correct, but the system was trained in a way to give a plausible duration for speech synthesis, in the aspect of a non-canonical spoken language.

