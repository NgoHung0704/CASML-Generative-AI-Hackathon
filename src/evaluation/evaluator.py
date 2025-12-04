"""
Evaluation module - Metrics calculation and scoring
"""

import numpy as np
from typing import List, Dict
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from bert_score import score as bert_score
import pandas as pd


class Evaluator:
    """
    Evaluates model predictions using various metrics.
    """
    
    def __init__(self, config):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.metrics = config.get('evaluation.metrics', ['bleu', 'rouge'])
        
        # Initialize ROUGE scorer
        rouge_types = config.get('evaluation.rouge.rouge_types', ['rouge1', 'rouge2', 'rougeL'])
        self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Evaluate predictions against references.
        
        Args:
            predictions: List of predicted answers
            references: List of ground truth answers
        
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        if 'bleu' in self.metrics:
            results.update(self._calculate_bleu(predictions, references))
        
        if 'rouge' in self.metrics:
            results.update(self._calculate_rouge(predictions, references))
        
        if 'bertscore' in self.metrics:
            results.update(self._calculate_bertscore(predictions, references))
        
        if 'exact_match' in self.metrics:
            results['exact_match'] = self._calculate_exact_match(predictions, references)
        
        return results
    
    def _calculate_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BLEU score"""
        # Wrap references in list for sacrebleu format
        refs = [[ref] for ref in references]
        
        bleu = corpus_bleu(predictions, list(zip(*refs)))
        
        return {
            'bleu': bleu.score,
            'bleu_precisions': bleu.precisions
        }
    
    def _calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        rouge_scores = {
            'rouge1_f': [],
            'rouge1_p': [],
            'rouge1_r': [],
            'rouge2_f': [],
            'rouge2_p': [],
            'rouge2_r': [],
            'rougeL_f': [],
            'rougeL_p': [],
            'rougeL_r': []
        }
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                if rouge_type in scores:
                    rouge_scores[f'{rouge_type}_f'].append(scores[rouge_type].fmeasure)
                    rouge_scores[f'{rouge_type}_p'].append(scores[rouge_type].precision)
                    rouge_scores[f'{rouge_type}_r'].append(scores[rouge_type].recall)
        
        # Average scores
        avg_scores = {k: np.mean(v) for k, v in rouge_scores.items() if v}
        
        return avg_scores
    
    def _calculate_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore"""
        bertscore_model = self.config.get('evaluation.bertscore.model', 'microsoft/deberta-xlarge-mnli')
        
        P, R, F1 = bert_score(predictions, references, model_type=bertscore_model, lang='en')
        
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }
    
    def _calculate_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """Calculate exact match accuracy"""
        matches = sum([1 for pred, ref in zip(predictions, references) 
                      if pred.strip().lower() == ref.strip().lower()])
        return matches / len(predictions) if predictions else 0.0
    
    def evaluate_single(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Evaluate a single prediction.
        
        Args:
            prediction: Predicted answer
            reference: Ground truth answer
        
        Returns:
            Dictionary of metric scores
        """
        return self.evaluate([prediction], [reference])


class SubmissionGenerator:
    """
    Generates submission file for Kaggle competition.
    """
    
    def __init__(self, config):
        """
        Initialize submission generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.submissions_dir = config.get('paths.submissions_dir')
    
    def generate_submission(
        self,
        question_ids: List[str],
        predictions: List[str],
        output_filename: str = 'submission.csv'
    ):
        """
        Generate submission CSV file.
        
        Args:
            question_ids: List of question IDs
            predictions: List of predicted answers
            output_filename: Output CSV filename
        """
        import os
        os.makedirs(self.submissions_dir, exist_ok=True)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'id': question_ids,
            'answer': predictions
        })
        
        # Save to CSV
        output_path = os.path.join(self.submissions_dir, output_filename)
        submission_df.to_csv(output_path, index=False)
        
        print(f"Submission saved to: {output_path}")
        
        return output_path
