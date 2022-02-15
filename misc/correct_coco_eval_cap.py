##########################################################
# Copyright 2019 Oath Inc.
# Licensed under the terms of the MIT license.
# Please see LICENSE file in the project root for terms.
##########################################################

from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

class CorrectCOCOEvalCap(COCOEvalCap):
    """This class inherits from COCOEvalCap in order to fix an issue
    with that class's implementation, without having to go into the
    coco-caption/ codebase. The COCOEvalCap implementation has a problem in
    how it assigns SPICE scores to the images, so that occasionally the
    computed SPICE scores end up getting assigned to the wrong images."""

    # This has to match the string for SPICE in COCOEvalCap, which is
    # hard-coded there.
    METHOD_SPICE = 'SPICE'

    # OVERRIDES: this function overrides setImgToEvalImgs() from COCOEvalCap,
    # in order to fix the ordering of the SPICE scores.
    def setImgToEvalImgs(self, scores, imgIds, method):
        if method == CorrectCOCOEvalCap.METHOD_SPICE:
            # The SPICE scores are actually ordered according the the sorted
            # imgIds, and not according the the original imgIds order.
            imgIds = sorted(imgIds)
        COCOEvalCap.setImgToEvalImgs(self, scores, imgIds, method)

    def evaluate(self, split: str):
        '''
            override the original evaluate method
        '''
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        if split == "train":
            scorers = [
                    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                    (Cider(), "CIDEr")
                ]
        else:
            scorers = [
                    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                    (Meteor(),"METEOR"),
                    (Rouge(), "ROUGE_L"),
                    (Cider(), "CIDEr"),
                    (Spice(), "SPICE")
                ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
        self.setEvalImgs()