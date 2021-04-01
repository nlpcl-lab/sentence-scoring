import tensorflow_hub as hub
from scipy import spatial


def get_model():
    """학습된 Universal Sentence Encoder(USE) 모델을 Tensorflow Hub에서 가져와서 반환합니다.

    Returns:
        (model): USE 모델 객체.
    """
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    return hub.load(module_url)


if __name__ == "__main__":
    sample_sentence1 = "This town is famous for its beautiful buildings."
    sample_sentence2 = "This town is famous for its awesome buildings."
    sample_sentence3 = "This town is not famous for its awful buildings."

    use_model = get_model()
    sent1_emb = use_model([sample_sentence1])[0]
    sent2_emb = use_model([sample_sentence2])[0]
    sent3_emb = use_model([sample_sentence3])[0]

    sent1_2_sim = 1 - spatial.distance.cosine(sent1_emb, sent2_emb)
    sent1_3_sim = 1 - spatial.distance.cosine(sent1_emb, sent3_emb)
    print("Sent1 and 2 similarity: {}".format(sent1_2_sim))
    print("Sent1 and 3 similarity: {}".format(sent1_3_sim))
    input(">>")
