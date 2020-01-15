import sklearn_crfsuite


def get_crf():
    """
    :return: a CRF learner with the specification used by medaCy
    """
    return sklearn_crfsuite.CRF(
                algorithm='l2sgd',
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True
            )
