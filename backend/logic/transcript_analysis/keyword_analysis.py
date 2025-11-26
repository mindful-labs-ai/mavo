from ekonlpy import Mecab
from collections import Counter, defaultdict
from soynlp.noun import LRNounExtractor_v2

def analyze_text_frequencies(texts, pos_tags_to_include=None):
    """
    Analyze frequency of words by POS tags in a list of texts.
    
    Args:
        texts (list): List of text strings to analyze
        pos_tags_to_include (list, optional): List of POS tag prefixes to include (e.g. ['NNG', 'VV']).
                                             If None, all tags will be included.
    
    Returns:
        dict: Dictionary where keys are POS tag prefixes and values are Counter objects with word frequencies
    """
    # Initialize the Mecab analyzer
    mecab = Mecab()
    
    # Initialize frequency counters for each POS tag
    tag_counters = defaultdict(Counter)
    
    # Process each text in the corpus
    for text in texts:
        # Perform POS tagging
        pos_tags = mecab.pos(text)
        
        # Iterate over the tagged words
        for word, tag in pos_tags:
            # Get the tag prefix (first few characters of the tag)
            tag_prefix = tag.split('+')[0]  # Handle compound tags
            
            # If specific tags are requested, check if this tag should be included
            if pos_tags_to_include is None or tag_prefix in pos_tags_to_include:
                tag_counters[tag_prefix][word] += 1
    
    return dict(tag_counters)

# Sample Korean text corpus
def extract_noun_frequencies_soynlp(texts, min_count: int = 2):
    """
    Extract nouns from texts using soynlp's LRNounExtractorV2 and return a Counter of noun frequencies.
    
    Args:
        texts (list): List of Korean sentences.
        min_count (int): Minimum frequency threshold passed to the extractor.
    
    Returns:
        Counter: Mapping noun → occurrence count across the corpus.
    """
    # 1) Train noun extractor & collect noun dict {noun: (count, score)}
    noun_extractor = LRNounExtractor_v2(verbose=False)
    nouns = noun_extractor.train_extract(texts)

    # 2) Build score dict for LTokenizer (use noun score)
    from soynlp.tokenizer import LTokenizer
    ltoken_scores = {noun: score for noun, (_, score) in nouns.items()}

    # 3) Initialise LTokenizer with these scores
    l_tokenizer = LTokenizer(ltoken_scores)

    # 4) Count only L‑tokens that are recognised nouns
    counter = Counter()
    for sent in texts:
        # LTokenizer returns token list based on L‑token rules
        tokens = l_tokenizer.tokenize(sent)
        for tok in tokens:
            if tok in nouns and nouns[tok][0] >= min_count:
                counter[tok] += 1

    return counter

def get_text_frequencies(texts, pos_tags=None):
    """
    Get frequencies of words by POS tags in a list of texts.
    
    Args:
        texts (list): List of text strings to analyze
        pos_tags (list, optional): List of POS tag prefixes to include.
                                  Default is ['NNG', 'NNP', 'VV'] (nouns and verbs).
    
    Returns:
        dict: Dictionary where keys are POS tag prefixes and values are Counter objects with word frequencies
    """
    if pos_tags is None:
        pos_tags = ['NNG', 'NNP', 'VV']
    
    frequencies = analyze_text_frequencies(texts, pos_tags)
    return frequencies

def format_word_frequencies(frequencies, pos_tags=None, top_n=5):
    """
    Format word frequencies by POS tags, adding '다' ending for verbs if not present.
    
    Args:
        frequencies (dict): Dictionary where keys are POS tag prefixes and values are Counter objects
        pos_tags (list, optional): List of POS tag prefixes to format. If None, all tags in frequencies will be used.
        top_n (int): Number of top words to include for each POS tag
    
    Returns:
        dict: Dictionary where keys are POS tag prefixes and values are lists of tuples (word, frequency)
               with properly formatted words
    """
    if pos_tags is None:
        pos_tags = list(frequencies.keys())
    
    formatted_results = {}
    
    for pos_tag in pos_tags:
        if pos_tag in frequencies:
            formatted_words = []
            for word, freq in frequencies[pos_tag].most_common(top_n):
                # Add '다' ending for verbs if not present
                if pos_tag == 'VV' and not word.endswith('다'):
                    word = word + '다'
                formatted_words.append((word, freq))
            formatted_results[pos_tag] = formatted_words
    
    return formatted_results

# Sample Korean text corpus
corpus = [
    "나는 오늘 아침에 커피를 마셨다.",
    "오늘은 날씨가 정말 좋다.",
    "커피를 마시면서 책을 읽었다.",
    "나는 커피를 좋아한다.",
    "요즘 기분이 우울하다.",
    "스트레스를 많이 받고 있다.",
    "행복한 순간을 떠올리면 기분이 좋아진다.",
    "불안한 마음이 계속된다.",
    "자신감을 잃은 것 같다.",
    "마음이 편안해지는 음악을 들었다.",
    "슬픈 영화를 보고 눈물이 났다.",
    "친구와의 대화가 위로가 되었다.",
    "혼자 있는 시간이 필요하다.",
    "감정을 표현하는 것이 어렵다.",
    "긍정적인 생각을 하려고 노력한다.",
    "불면증 때문에 밤새 뒤척였다.",
    "일에 대한 의욕이 사라졌다.",
    "사소한 일에도 화가 난다.",
    "마음의 여유를 찾고 싶다.",
    "자존감이 낮아진 것 같다.",
    "새로운 취미를 시작해봤다.",
    "운동을 하면 기분이 좋아진다.",
    "명상을 통해 마음을 다스린다.",
    "과거의 상처가 떠오른다.",
    "미래에 대한 걱정이 많다.",
    "사랑하는 사람과의 이별이 힘들다.",
    "자신을 이해해주는 사람이 필요하다.",
    "감정을 글로 표현해본다.",
    "심리 상담을 받아볼까 고민 중이다.",
    "마음속에 쌓인 감정을 털어놓고 싶다.",
    "일상에서 작은 행복을 찾으려 한다.",
    "자연 속에서 힐링을 느낀다.",
    "혼자 여행을 떠나고 싶다.",
    "새로운 사람들과의 만남이 두렵다.",
    "과거의 실수를 후회한다.",
    "자신을 용서하는 법을 배우고 싶다.",
    "마음의 상처를 치유하고 싶다.",
    "스트레스를 해소할 방법을 찾고 있다.",
    "감정을 억누르지 않고 표현하려 한다.",
    "내면의 평화를 찾고 싶다.",
    "심리적인 안정이 필요하다.",
    "자신을 사랑하는 법을 배우고 있다.",
    "마음의 짐을 내려놓고 싶다.",
    "긍정적인 에너지를 받고 싶다.",
    "자신의 감정을 인정하려 노력한다.",
    "심리적인 성장을 이루고 싶다.",
    "감정의 기복이 심하다.",
    "마음이 공허하게 느껴진다.",
    "자신의 감정을 이해하려 한다.",
    "심리적인 지지를 받고 싶다.",
    "마음속의 불안을 해소하고 싶다.",
    "니가 가라 하와이.",
    "살아있네!",
    "어이가 없네.",
    "밥은 먹고 다니냐?",
    "오늘부터 1일이다.",
    "괜찮아, 다 잘 될 거야.",
    "넌 계획이 다 있구나.",
    "그만해라, 이 악마야!",
    "모히토 가서 몰디브 한 잔?",
    "이거 실화냐?",
    "형, 나 죽어!",
    "살려는 드릴게.",
    "제발 좀 하지 마!",
    "무슨 일이 있어도 우리가 원하는 걸 얻어.",
    "나한테 왜 그랬어요?",
    "철수야, 오늘 저녁에 홍대에서 볼래?",
    "지민아, 카카오톡 확인했어?",
    "우리 내일 강남역 스타벅스에서 회의하자.",
    "지은이 생일파티 준비 다 했어?",
    "태현이 축구 경기 보러 잠실 종합운동장 갈래?",
    "영희야, 서울대 후문 앞 카페 알지?",
    "동훈 씨, 삼성동 코엑스에서 점심 어때요?",
    "소연아, 네이버 지도 좀 열어봐.",
    "민수야, 다음 주 월요일에 현대백화점 앞에서 만나자.",
    "지훈이랑 수빈이 영화 '기생충' 다시 보자.",
    "서연아, 오늘 카페24 세미나 신청했어?",
    "정민 씨, 부산 해운대 출장 가본 적 있어요?",
    "해진아, 오늘 오후에 롯데월드 갈까?",
    "윤호야, 이마트 트레이더스 신상품 나온 거 봤어?",
    "보라야, 카카오뱅크 계좌번호 좀 보내줄래?",
]

# Example usage
if __name__ == "__main__":
    # Analyze only nouns and verbs
    # @ref pos types: https://docs.google.com/spreadsheets/d/1OGAjUvalBuX-oZvZ_-9tEfYD2gQe7hTGsgUpiiBSXI8/edit?pli=1&gid=0#gid=0
    list_pos_tags = ['NNG', 'NNP', 'VV']
    
    # Get the raw frequencies
    frequencies = get_text_frequencies(corpus, list_pos_tags)
    
    # Format the results with proper verb endings
    formatted_results = format_word_frequencies(frequencies, list_pos_tags)
    
    # Display the formatted results
    for pos_tag, words in formatted_results.items():
        print(f"\nTop {pos_tag}:")
        for word, freq in words:
                print(f"{word}: {freq}")

    # ---- soynlp noun extraction ----
    noun_freqs_soynlp = extract_noun_frequencies_soynlp(corpus, min_count=2)

    print("\nTop Nouns by soynlp:")
    for noun, freq in noun_freqs_soynlp.most_common(5):
        print(f"{noun}: {freq}")