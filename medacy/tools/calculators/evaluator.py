"""Inter-annotator agreement calculator."""

import glob
import os
from collections import defaultdict
from xml.etree import cElementTree


class ClinicalCriteria(object):
    """Criteria in the Track 1 documents."""

    def __init__(self, tid, value):
        """Init."""
        self.tid = tid.strip().upper()
        self.ttype = self.tid
        self.value = value.lower().strip()

    def equals(self, other, mode='strict'):
        """Return whether the current criteria is equal to the one provided."""
        if other.tid == self.tid and other.value == self.value: return True
        return False


class ClinicalConcept(object):
    """Named Entity Tag class."""

    def __init__(self, tid, start, end, ttype, text=''):
        """Init."""
        self.tid = str(tid).strip()
        self.start = int(start)
        self.end = int(end)
        self.text = str(text).strip()
        self.ttype = str(ttype).strip()

    def span_matches(self, other, mode='strict'):
        """Return whether the current tag overlaps with the one provided."""
        if mode not in ('strict', 'lenient'): raise ValueError("mode must be 'strict' or 'lenient'.")
        if mode == 'strict':
            if self.start == other.start and self.end == other.end:
                return True
        else:   # lenient
            if (self.end > other.start and self.start < other.end) or \
                    (self.start < other.end and other.start < self.end):
                return True
        return False

    def equals(self, other, mode='strict'):
        """Return whether the current tag is equal to the one provided."""
        if mode not in ('strict', 'lenient'): raise ValueError("mode must be 'strict' or 'lenient'.")
        return other.ttype == self.ttype and self.span_matches(other, mode)

    def __str__(self):
        """String representation."""
        return '{}\t{}\t({}:{})'.format(self.ttype, self.text, self.start, self.end)


class Relation(object):
    """Relation class."""

    def __init__(self, rid, arg1, arg2, rtype):
        """Init."""
        assert isinstance(arg1, ClinicalConcept)
        assert isinstance(arg2, ClinicalConcept)
        self.rid = str(rid).strip()
        self.arg1 = arg1
        self.arg2 = arg2
        self.rtype = str(rtype).strip()

    def equals(self, other, mode='strict'):
        """Return whether the current tag is equal to the one provided."""
        if mode not in ('strict', 'lenient'): raise ValueError("mode must be 'strict' or 'lenient'.")
        if self.arg1.equals(other.arg1, mode) and \
                self.arg2.equals(other.arg2, mode) and \
                self.rtype == other.rtype:
            return True
        return False

    def __str__(self):
        """String representation."""
        return '{} ({}->{})'.format(self.rtype, self.arg1.ttype, self.arg2.ttype)


class RecordTrack1(object):
    """Record for Track 2 class."""

    def __init__(self, file_path):
        self.path = os.path.abspath(file_path)
        self.basename = os.path.basename(self.path)
        self.annotations = self._get_annotations()
        self.text = None

    @property
    def tags(self):
        return self.annotations['tags']

    def _get_annotations(self):
        """Return a dictionary with all the annotations in the .ann file."""
        annotations = defaultdict(dict)
        annotation_file = cElementTree.parse(self.path)
        for tag in annotation_file.findall('.//TAGS/*'):
            criterion = ClinicalCriteria(tag.tag.upper(), tag.attrib['met'])
            annotations['tags'][tag.tag.upper()] = criterion
            if tag.attrib['met'] not in ('met', 'not met'):
                assert '{}: Unexpected value ("{}") for the {} tag!'.format(
                    self.path, criterion.value, criterion.ttype)
        return annotations


class RecordTrack2(object):
    """Record for Track 2 class."""

    def __init__(self, file_path):
        """Initialize."""
        self.path = os.path.abspath(file_path)
        self.basename = os.path.basename(self.path)
        self.annotations = self._get_annotations()
        # self.text = self._get_text()

    @property
    def tags(self):
        return self.annotations['tags']

    @property
    def relations(self):
        return self.annotations['relations']

    def _get_annotations(self):
        """Return a dictionary with all the annotations in the .ann file."""
        annotations = defaultdict(dict)
        with open(self.path) as annotation_file:
            lines = annotation_file.readlines()

        for line_num, line in enumerate(lines):
            if line.strip().startswith('T'):
                try:
                    tag_id, tag_m, tag_text = line.strip().split('\t')
                except ValueError:
                    print(self.path, line)
                if len(tag_m.split(' ')) == 3:
                    tag_type, tag_start, tag_end = tag_m.split(' ')
                elif len(tag_m.split(' ')) == 4:
                    tag_type, tag_start, _, tag_end = tag_m.split(' ')
                elif len(tag_m.split(' ')) == 5:
                    tag_type, tag_start, _, _, tag_end = tag_m.split(' ')
                else:
                    print(self.path)
                    print(line)
                tag_start, tag_end = int(tag_start), int(tag_end)
                annotations['tags'][tag_id] = ClinicalConcept(tag_id, tag_start, tag_end, tag_type, tag_text)
            if line.strip().startswith('R'):
                rel_id, rel_m = line.strip().split('\t')
                rel_type, rel_arg1, rel_arg2 = rel_m.split(' ')
                rel_arg1 = rel_arg1.split(':')[1]
                rel_arg2 = rel_arg2.split(':')[1]
                arg1 = annotations['tags'][rel_arg1]
                arg2 = annotations['tags'][rel_arg2]
                annotations['relations'][rel_id] = Relation(rel_id, arg1, arg2, rel_type)

        return annotations

    def _get_text(self):
        """Return the text in the corresponding txt file."""
        path = self.path.replace('.ann', '.txt')
        with open(path) as text_file:
            text = text_file.read()
        return text

    def search_by_id(self, key):
        """Search by id among both tags and relations."""
        try:
            return self.annotations['tags'][key]
        except KeyError():
            try:
                return self.annotations['relations'][key]
            except KeyError():
                return None


class Measures(object):
    """Abstract methods and var to evaluate."""

    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        """Initizialize."""
        for p in (tp, tn, fp, fn):
            if not (isinstance(p, int)): raise TypeError("%s must be of class int" % p)

        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def precision(self):
        """Compute Precision score."""
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 0.0

    def recall(self):
        """Compute Recall score."""
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 0.0

    def f_score(self, beta=1):
        """Compute F1-measure score."""
        assert beta > 0.
        try:
            num = (1 + beta**2) * (self.precision() * self.recall())
            den = beta**2 * (self.precision() + self.recall())
            return num / den
        except ZeroDivisionError:
            return 0.0

    def f1(self):
        """Compute the F1-score (beta=1)."""
        return self.f_score(beta=1)

    def specificity(self):
        """Compute Specificity score."""
        try:
            return self.tn / (self.fp + self.tn)
        except ZeroDivisionError:
            return 0.0

    def sensitivity(self):
        """Compute Sensitivity score."""
        return self.recall()

    def auc(self):
        """Compute AUC score."""
        return (self.sensitivity() + self.specificity()) / 2


class SingleEvaluator(object):
    """Evaluate two single files."""

    def __init__(self, doc1, doc2, track, mode='strict', key=None, verbose=False):
        """Initialize."""
        assert isinstance(doc1, RecordTrack2) or isinstance(doc1, RecordTrack1)
        assert isinstance(doc2, RecordTrack2) or isinstance(doc2, RecordTrack1)
        if mode not in ('strict', 'lenient'): raise ValueError("mode must be 'strict' or 'lenient'.")
        assert doc1.basename == doc2.basename
        self.scores = {
            'tags': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            'relations': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        }
        self.doc1 = doc1
        self.doc2 = doc2
        if key:
            gol = [t for t in doc1.tags.values() if t.ttype == key]
            sys = [t for t in doc2.tags.values() if t.ttype == key]
            sys_check = [t for t in doc2.tags.values() if t.ttype == key]
        else:
            gol = [t for t in doc1.tags.values()]
            sys = [t for t in doc2.tags.values()]
            sys_check = [t for t in doc2.tags.values()]

        #pare down matches -- if multiple system tags overlap with only one
        #gold standard tag, only keep one sys tag
        gol_matched = []
        for s in sys:
            for g in gol:
                if (g.equals(s,mode)):
                    if g not in gol_matched:
                        gol_matched.append(g)
                    else:
                        if s in sys_check:
                            sys_check.remove(s)


        sys = sys_check
        #now evaluate
        self.scores['tags']['tp'] = len({s.tid for s in sys for g in gol if g.equals(s, mode)})
        self.scores['tags']['fp'] = len({s.tid for s in sys}) - self.scores['tags']['tp']
        self.scores['tags']['fn'] = len({g.tid for g in gol}) - self.scores['tags']['tp']
        self.scores['tags']['tn'] = 0

        if verbose and track == 2:
            tps = {s for s in sys for g in gol if g.equals(s, mode)}
            fps = set(sys) - tps
            fns = set()
            for g in gol:
                if not len([s for s in sys if s.equals(g, mode)]):
                    fns.add(g)
            for e in fps:
                print('FP: ' + str(e))
            for e in fns:
                print('FN:' + str(e))
        if track == 2:
            if key:
                gol = [r for r in doc1.relations.values() if r.rtype == key]
                sys = [r for r in doc2.relations.values() if r.rtype == key]
                sys_check = [r for r in doc2.relations.values() if r.rtype == key]
            else:
                gol = [r for r in doc1.relations.values()]
                sys = [r for r in doc2.relations.values()]
                sys_check = [r for r in doc2.relations.values()]

            # pare down matches -- if multiple system tags overlap with only one
            # gold standard tag, only keep one sys tag
            gol_matched = []
            for s in sys:
                for g in gol:
                    if (g.equals(s,mode)):
                        if g not in gol_matched:
                            gol_matched.append(g)
                        else:
                            if s in sys_check:
                                sys_check.remove(s)
            sys = sys_check
            # now evaluate
            self.scores['relations']['tp'] = len({s.rid for s in sys for g in gol if g.equals(s, mode)})
            self.scores['relations']['fp'] = len({s.rid for s in sys}) - self.scores['relations']['tp']
            self.scores['relations']['fn'] = len({g.rid for g in gol}) - self.scores['relations']['tp']
            self.scores['relations']['tn'] = 0
            if verbose:
                tps = {s for s in sys for g in gol if g.equals(s, mode)}
                fps = set(sys) - tps
                fns = set()
                for g in gol:
                    if not len([s for s in sys if s.equals(g, mode)]):
                        fns.add(g)
                for e in fps:
                    print('FP: ' + str(e))
                for e in fns:
                    print('FN:' + str(e))


class MultipleEvaluator(object):
    """Evaluate two sets of files."""

    def __init__(self, corpora, tags=(), relations=(), tag_type=None, mode='strict', verbose=False):
        """Initialize."""
        assert isinstance(corpora, Corpora)
        if mode not in ('strict', 'lenient'): raise ValueError("mode must be 'strict' or 'lenient'.")

        self.scores = None

        if corpora.track == 1:
            self.track1(corpora)
        else:
            self.track2(corpora, tag_type, mode, verbose)

        if not isinstance(tags, tuple):
            self.tags = tuple(tags)
        if not isinstance(relations, tuple):
            self.relations = tuple(relations)


    def track1(self, corpora):
        """Compute measures for Track 1."""
        self.scores = defaultdict(dict)
        metrics = ('p', 'r', 'f1', 'specificity', 'auc')
        values = ('met', 'not met')
        self.values = {
            'met': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
            'not met': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        }

        def evaluation(corpora, value, scores):
            predictions = defaultdict(list)
            for g, s in corpora.docs:
                for tag in self.tags:
                    predictions[tag].append(
                        (g.tags[tag].value == value, s.tags[tag].value == value))
            for tag in self.tags:
                # accumulate for micro overall measure
                self.values[value]['tp'] += predictions[tag].count((True, True))
                self.values[value]['fp'] += predictions[tag].count((False, True))
                self.values[value]['tn'] += predictions[tag].count((False, False))
                self.values[value]['fn'] += predictions[tag].count((True, False))

                # compute per-tag measures
                measures = Measures(
                    tp=predictions[tag].count((True, True)),
                    fp=predictions[tag].count((False, True)),
                    tn=predictions[tag].count((False, False)),
                    fn=predictions[tag].count((True, False))
                )
                scores[(tag, value, 'p')] = measures.precision()
                scores[(tag, value, 'r')] = measures.recall()
                scores[(tag, value, 'f1')] = measures.f1()
                scores[(tag, value, 'specificity')] = measures.specificity()
                scores[(tag, value, 'auc')] = measures.auc()
            return scores

        self.scores = evaluation(corpora, 'met', self.scores)
        self.scores = evaluation(corpora, 'not met', self.scores)

        for measure in metrics:
            for value in values:
                self.scores[('macro', value, measure)] = sum(
                    [self.scores[(t, value, measure)] for t in self.tags]) / len(self.tags)

    def track2(self, corpora, tag_type=None, mode='strict', verbose=False):
        """Compute measures for Track 2."""
        self.scores = {'tags': {'tp': 0,
                                'fp': 0,
                                'fn': 0,
                                'tn': 0,
                                'micro': {'precision': 0,
                                          'recall': 0,
                                          'f1': 0},
                                'macro': {'precision': 0,
                                          'recall': 0,
                                          'f1': 0}},
                       'relations': {'tp': 0,
                                     'fp': 0,
                                     'fn': 0,
                                     'tn': 0,
                                     'micro': {'precision': 0,
                                               'recall': 0,
                                               'f1': 0},
                                     'macro': {'precision': 0,
                                               'recall': 0,
                                               'f1': 0}}}

        for g, s in corpora.docs:
            evaluator = SingleEvaluator(g, s, 2, mode, tag_type, verbose=verbose)
            for target in ('tags', 'relations'):
                for score in ('tp', 'fp', 'fn'):
                    self.scores[target][score] += evaluator.scores[target][score]
                measures = Measures(tp=evaluator.scores[target]['tp'],
                                    fp=evaluator.scores[target]['fp'],
                                    fn=evaluator.scores[target]['fn'],
                                    tn=evaluator.scores[target]['tn'])
                for score in ('precision', 'recall', 'f1'):
                    fn = getattr(measures, score)
                    self.scores[target]['macro'][score] += fn()

        for target in ('tags', 'relations'):
            # Normalization
            for key in self.scores[target]['macro'].keys():
                self.scores[target]['macro'][key] = self.scores[target]['macro'][key] / len(corpora.docs)

            measures = Measures(
                tp=self.scores[target]['tp'],
                fp=self.scores[target]['fp'],
                fn=self.scores[target]['fn'],
                tn=self.scores[target]['tn']
            )
            for key in self.scores[target]['micro'].keys():
                fn = getattr(measures, key)
                self.scores[target]['micro'][key] = fn()


def evaluate(corpora, entities, relations, mode='strict', verbose=False):
    """Run the evaluation by considering only files in the two folders."""
    if mode not in ('strict', 'lenient'): raise ValueError("mode must be 'strict' or 'lenient'.")
    evaluator_s = MultipleEvaluator(corpora, tags=entities, relations=relations, verbose=verbose)  # TODO check
    evaluator_l = MultipleEvaluator(corpora, tags=entities, relations=relations, mode='lenient', verbose=verbose)
    output_str = "Entities\n"
    output_str += "\t\tPrec.\tRec.\tF_1\n"

    for tag in set(evaluator_s.tags):
        evaluator_tag_s = MultipleEvaluator(corpora, tags=tag, verbose=verbose)
        evaluator_tag_l = MultipleEvaluator(corpora, tags=tag, mode='lenient', verbose=verbose)
        output_str += 'Strict\t{}\t{:<5.4f}\t{:<5.4f}\t{:<5.4f}\nLenient\t{}\t{:<5.4f}\t{:<5.4f}\t{:<5.4f}\n'.format(
            tag.capitalize(),
            evaluator_tag_s.scores['tags']['micro']['precision'],
            evaluator_tag_s.scores['tags']['micro']['recall'],
            evaluator_tag_s.scores['tags']['micro']['f1'],
            tag.capitalize(),
            evaluator_tag_l.scores['tags']['micro']['precision'],
            evaluator_tag_l.scores['tags']['micro']['recall'],
            evaluator_tag_l.scores['tags']['micro']['f1']
        )

    output_str += "Strict\tOverall (micro)\t{:<5.4f}\t{:<5.4f}\t{:<5.4f}\nLenient\tOverall (macro)\t{:<5.4f}\t{:<5.4f}\t{:<5.4f}\n".format(
        evaluator_s.scores['tags']['micro']['precision'],
        evaluator_s.scores['tags']['micro']['recall'],
        evaluator_s.scores['tags']['micro']['f1'],
        evaluator_l.scores['tags']['micro']['precision'],
        evaluator_l.scores['tags']['micro']['recall'],
        evaluator_l.scores['tags']['micro']['f1']
    )

    output_str += 'Strict\tOverall (macro)\t{:<5.4f}\t{:<5.4f}\t{:<5.4f}\nLenient\tOverall (micro)\t{:<5.4f}\t{:<5.4f}\t{:<5.4f}\n'.format(
        evaluator_s.scores['tags']['macro']['precision'],
        evaluator_s.scores['tags']['macro']['recall'],
        evaluator_s.scores['tags']['macro']['f1'],
        evaluator_l.scores['tags']['macro']['precision'],
        evaluator_l.scores['tags']['macro']['recall'],
        evaluator_l.scores['tags']['macro']['f1']
    )

    output_str += "Relations\n"
    for rel in evaluator_s.relations:
        evaluator_tag_s = MultipleEvaluator(corpora, relations=rel, mode='strict', verbose=verbose)
        evaluator_tag_l = MultipleEvaluator(corpora, relations=rel, mode='lenient', verbose=verbose)
        output_str += 'Strict\t{}\t{:<5.4f}\t{:<5.4f}\t{:<5.4f}\nLenient\t{}\t{:<5.4f}\t{:<5.4f}\t{:<5.4f}\n'.format(
            '{} -> {}'.format(rel.split('-')[0], rel.split('-')[1].capitalize()),
            evaluator_tag_s.scores['relations']['micro']['precision'],
            evaluator_tag_s.scores['relations']['micro']['recall'],
            evaluator_tag_s.scores['relations']['micro']['f1'],
            '{} -> {}'.format(rel.split('-')[0], rel.split('-')[1].capitalize()),
            evaluator_tag_l.scores['relations']['micro']['precision'],
            evaluator_tag_l.scores['relations']['micro']['recall'],
            evaluator_tag_l.scores['relations']['micro']['f1']
        )

    output_str += 'Strict\tOverall (micro)\t{:<5.4f}\t{:<5.4f}\t{:<5.4f}\nLenient\tOverall (micro)\t{:<5.4f}\t{:<5.4f}\t{:<5.4f}\n'.format(
        evaluator_s.scores['relations']['micro']['precision'],
        evaluator_s.scores['relations']['micro']['recall'],
        evaluator_s.scores['relations']['micro']['f1'],
        evaluator_l.scores['relations']['micro']['precision'],
        evaluator_l.scores['relations']['micro']['recall'],
        evaluator_l.scores['relations']['micro']['f1']
    )
    output_str += 'Strict\tOverall (macro)\t{:<5.4f}\t{:<5.4f}\t{:<5.4f}\nLenient\tOverall (macro)\t{:<5.4f}\t{:<5.4f}\t{:<5.4f}\n'.format(
        evaluator_s.scores['relations']['macro']['precision'],
        evaluator_s.scores['relations']['macro']['recall'],
        evaluator_s.scores['relations']['macro']['f1'],
        evaluator_l.scores['relations']['macro']['precision'],
        evaluator_l.scores['relations']['macro']['recall'],
        evaluator_l.scores['relations']['macro']['f1']
    )
    print('{:20}{:^48}'.format('', '  {} files found  '.format(len(corpora.docs))))
    print(output_str)
    return output_str


class Corpora(object):

    def __init__(self, folder1, folder2, track_num):
        extensions = {1: '*.xml', 2: '*.ann'}
        file_ext = extensions[track_num]
        self.track = track_num
        self.folder1 = folder1
        self.folder2 = folder2
        files1 = set([os.path.basename(f) for f in glob.glob(
            os.path.join(folder1, file_ext))])
        files2 = set([os.path.basename(f) for f in glob.glob(
            os.path.join(folder2, file_ext))])
        common_files = files1 & files2     # intersection
        if not common_files:
            print('ERROR: None of the files match.')
        else:
            if files1 - common_files:
                print('Files skipped in {}:'.format(self.folder1))
                print(', '.join(sorted(list(files1 - common_files))))
            if files2 - common_files:
                print('Files skipped in {}:'.format(self.folder2))
                print(', '.join(sorted(list(files2 - common_files))))
        self.docs = []
        for file in common_files:
            if track_num == 1:
                g = RecordTrack1(os.path.join(self.folder1, file))
                s = RecordTrack1(os.path.join(self.folder2, file))
            else:
                g = RecordTrack2(os.path.join(self.folder1, file))
                s = RecordTrack2(os.path.join(self.folder2, file))
            self.docs.append((g, s))


def evaluate_annotation_agreement(predicted_path, gold_path, entities: list, relations: list, track=2, verbose=False):
    """
    Prints the comparison between two datasets to the console
    :param predicted_path: Directory containing the predicted dataset
    :param gold_path: Directory containing the gold dataset
    :param entities: The list of entities you want to calculate for
    :param relations: The list of relations you want to calculate for
    :param track:
    :param verbose: Whether or not to be verbose
    :return: None
    """
    corpora = Corpora(gold_path, predicted_path, track)
    if corpora.docs:
        return evaluate(corpora, entities=entities, relations=relations, verbose=verbose)


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser(description='Inter-annotation agreement calculator.')
    parser.add_argument('prediction_folder', help='Predicted data files directory path')
    parser.add_argument('gold_folder', help='Gold data files directory path')
    parser.add_argument('config', help="Path to a JSON for entities and relations")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.loads(f.read())

    entities = config["entities"]
    relations = config["relations"]

    evaluate_annotation_agreement(
        predicted_path=os.path.abspath(args.predicted_folder),
        gold_path=os.path.abspath(args.gold_folder),
        entities=entities,
        relations=relations
    )