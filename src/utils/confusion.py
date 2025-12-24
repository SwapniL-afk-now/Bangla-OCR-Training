import collections
from typing import List, Dict

class CharacterConfusionMatrix:
    def __init__(self, max_size=50):
        self.confusion_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        self.total_errors = 0
        self.max_size = max_size
    
    def update(self, predictions: List[str], references: List[str]):
        for pred, ref in zip(predictions, references):
            pred = pred.strip()
            ref = ref.strip()
            
            max_len = max(len(pred), len(ref))
            for i in range(max_len):
                ref_char = ref[i] if i < len(ref) else '<EOS>'
                pred_char = pred[i] if i < len(pred) else '<EOS>'
                
                if ref_char != pred_char:
                    self.confusion_counts[ref_char][pred_char] += 1
                    self.total_errors += 1
    
    def get_top_confusions(self, top_k=10):
        all_confusions = []
        for true_char, pred_dict in self.confusion_counts.items():
            for pred_char, count in pred_dict.items():
                all_confusions.append((true_char, pred_char, count))
        
        all_confusions.sort(key=lambda x: x[2], reverse=True)
        return all_confusions[:top_k]
    
    def get_problematic_chars(self, threshold=10):
        problematic = {}
        for true_char, pred_dict in self.confusion_counts.items():
            total_errors_for_char = sum(pred_dict.values())
            if total_errors_for_char >= threshold:
                problematic[true_char] = total_errors_for_char
        
        return dict(sorted(problematic.items(), key=lambda x: x[1], reverse=True))
    
    def print_summary(self, step, config_threshold=10):
        top_confusions = self.get_top_confusions(10)
        problematic = self.get_problematic_chars(config_threshold)
        
        print(f"\n{'='*70}")
        print(f"CHARACTER CONFUSION ANALYSIS - Step {step}")
        print(f"{'='*70}")
        print(f"Total character errors: {self.total_errors}")
        print(f"\nTop 10 Confusions:")
        print(f"{'True':<10} {'Pred':<10} {'Count':<10}")
        print("-" * 35)
        for true_char, pred_char, count in top_confusions:
            print(f"{true_char:<10} {pred_char:<10} {count:<10}")
        
        if problematic:
            print(f"\nProblematic Characters (>{config_threshold} errors):")
            for char, error_count in list(problematic.items())[:15]:
                print(f"  '{char}': {error_count} errors")
        
        print(f"{'='*70}\n")
