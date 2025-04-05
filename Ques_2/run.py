import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class IndianLanguageAudioAnalyzer:
    def __init__(self, data_dir, languages=['Hindi', 'Tamil', 'Bengali'], n_mfcc=13):
        """
        Initialize the analyzer for Indian languages audio.
        """
        self.data_dir = data_dir
        self.languages = languages
        self.n_mfcc = n_mfcc
        self.language_data = {}
        self.file_paths = self._collect_file_paths()
        # Cache for processed audio data: maps file_path -> processed result.
        self.audio_cache = {}

    def _collect_file_paths(self):
        """
        Collect all audio file paths for each language.
        """
        file_paths = {lang: [] for lang in self.languages}
        
        print("Collecting file paths for each language...")
        for language in self.languages:
            language_dir = os.path.join(self.data_dir, language)
            if not os.path.exists(language_dir):
                print(f"Warning: Directory not found for {language}")
                continue
                
            files = os.listdir(language_dir)
            audio_files = [os.path.join(language_dir, f) for f in files 
                           if f.endswith('.wav') or f.endswith('.mp3')]
            file_paths[language] = audio_files
            print(f"Found {len(audio_files)} audio files for {language}.")
            
        return file_paths
        
    def load_audio(self, file_path, duration=3.0):
        """
        Load audio file, convert to mono, and normalize.
        """
        try:
            y, sr = librosa.load(file_path, sr=None, duration=duration, mono=True)
            if y is not None and len(y) > 0:
                y = librosa.util.normalize(y)
            return y, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def extract_mfcc(self, y, sr):
        if y is None:
            return None, None, None
    
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc, mode='constant')
        delta2_mfcc = librosa.feature.delta(mfcc, order=2, mode='constant')
    
        return mfcc, delta_mfcc, delta2_mfcc

    def process_audio_files(self, file_paths=None, max_files_per_language=None):
        """
        Process audio files and extract features.
        Uses a cache to avoid re-processing files.
        """
        processed_data = []
        
        if file_paths is not None:
            print("Processing provided file paths...")
            for file_path in tqdm(file_paths, desc="Processing files"):
                # If cached, use the saved result
                if file_path in self.audio_cache:
                    processed_data.append(self.audio_cache[file_path])
                    continue
                    
                language = None
                for lang in self.languages:
                    if lang in file_path:
                        language = lang
                        break
                if language is None:
                    print(f"Warning: Could not determine language for {file_path}")
                    continue
                    
                y, sr = self.load_audio(file_path)
                if y is not None:
                    mfcc_features, delta_mfcc, delta2_mfcc = self.extract_mfcc(y, sr)
                    if mfcc_features is not None:
                        data_item = {
                            'file': file_path,
                            'language': language,
                            'mfcc': mfcc_features,
                            'delta_mfcc': delta_mfcc,
                            'delta2_mfcc': delta2_mfcc,
                            'sr': sr
                        }
                        processed_data.append(data_item)
                        self.audio_cache[file_path] = data_item
        else:
            print("Processing files by language...")
            for language, lang_files in self.file_paths.items():
                print(f"Processing {language} audio files...")
                if max_files_per_language is not None:
                    lang_files = lang_files[:max_files_per_language]
                for file_path in tqdm(lang_files, desc=f"Processing {language} files"):
                    if file_path in self.audio_cache:
                        processed_data.append(self.audio_cache[file_path])
                        continue
                    y, sr = self.load_audio(file_path)
                    if y is not None:
                        mfcc_features, delta_mfcc, delta2_mfcc = self.extract_mfcc(y, sr)
                        if mfcc_features is not None:
                            data_item = {
                                'file': file_path,
                                'language': language,
                                'mfcc': mfcc_features,
                                'delta_mfcc': delta_mfcc,
                                'delta2_mfcc': delta2_mfcc,
                                'sr': sr
                            }
                            processed_data.append(data_item)
                            self.audio_cache[file_path] = data_item
                count = len([data for data in processed_data if data['language'] == language])
                print(f"Processed {count} files for {language}.")
        
        return processed_data
    
    def visualize_mfcc_spectrograms(self, processed_data, num_samples=3):
        """
        Visualize MFCC spectrograms for each language.
        """
        language_groups = {}
        for item in processed_data:
            lang = item['language']
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append(item)
            
        for language, samples in language_groups.items():
            vis_samples = samples[:num_samples]
            fig, axes = plt.subplots(len(vis_samples), 1, figsize=(12, 4 * len(vis_samples)))
            if len(vis_samples) == 1:
                axes = [axes]
                
            for i, sample in enumerate(vis_samples):
                img = librosa.display.specshow(
                    sample['mfcc'], 
                    x_axis='time', 
                    ax=axes[i],
                    sr=sample['sr']
                )
                axes[i].set_title(f'{language} - Sample {i+1} MFCC')
                plt.colorbar(img, ax=axes[i], format='%+2.0f dB')
            
            plt.tight_layout()
            plt.savefig(f'{language}_mfcc_spectrograms.png')
            plt.close()
            print(f"Saved MFCC spectrograms for {language}.")
    
    def compute_statistics(self, processed_data):
        """
        Compute statistics for MFCC coefficients across languages.
        """
        stats = []
        language_groups = {}
        for item in processed_data:
            lang = item['language']
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append(item)
        
        for language, samples in language_groups.items():
            all_mfccs = np.concatenate([sample['mfcc'] for sample in samples], axis=1)
            for i in range(self.n_mfcc):
                coef = all_mfccs[i, :]
                stats.append({
                    'Language': language,
                    'Coefficient': i,
                    'Mean': np.mean(coef),
                    'Median': np.median(coef),
                    'Std': np.std(coef),
                    'Min': np.min(coef),
                    'Max': np.max(coef),
                    '25th': np.percentile(coef, 25),
                    '75th': np.percentile(coef, 75)
                })
        stats_df = pd.DataFrame(stats)
        print("Computed statistics for MFCC coefficients.")
        return stats_df
    
    def visualize_statistics(self, stats_df):
        """
        Visualize statistics across languages.
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Coefficient', y='Mean', hue='Language', data=stats_df)
        plt.title('Mean of MFCC Coefficients by Language')
        plt.xlabel('MFCC Coefficient')
        plt.ylabel('Mean Value')
        plt.savefig('mfcc_coefficient_means.png')
        plt.close()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Coefficient', y='Std', hue='Language', data=stats_df)
        plt.title('Standard Deviation of MFCC Coefficients by Language')
        plt.xlabel('MFCC Coefficient')
        plt.ylabel('Standard Deviation')
        plt.savefig('mfcc_coefficient_std.png')
        plt.close()
        
        top_coeffs = stats_df.groupby('Coefficient')['Std'].mean().nlargest(5).index
        top_coeff_data = stats_df[stats_df['Coefficient'].isin(top_coeffs)]
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Coefficient', y='Mean', hue='Language', data=top_coeff_data)
        plt.title('Distribution of Top 5 MFCC Coefficients by Language')
        plt.savefig('top_coefficients_boxplot.png')
        plt.close()
        print("Visualized statistics for MFCC coefficients.")
    
    def perform_statistical_tests(self, processed_data):
        """
        Perform statistical tests to compare languages.
        """
        language_groups = {}
        for item in processed_data:
            lang = item['language']
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append(item)
        
        test_results = {}
        for i, lang1 in enumerate(self.languages):
            for j, lang2 in enumerate(self.languages[i+1:], i+1):
                if lang1 not in language_groups or lang2 not in language_groups:
                    continue
                    
                pair_key = f"{lang1}_vs_{lang2}"
                test_results[pair_key] = []
                
                lang1_mfccs = [sample['mfcc'] for sample in language_groups[lang1]]
                lang2_mfccs = [sample['mfcc'] for sample in language_groups[lang2]]
                
                for coef in range(self.n_mfcc):
                    lang1_vals = np.concatenate([mfcc[coef, :] for mfcc in lang1_mfccs])
                    lang2_vals = np.concatenate([mfcc[coef, :] for mfcc in lang2_mfccs])
                    t_stat, p_val = ttest_ind(lang1_vals, lang2_vals, equal_var=False)
                    
                    test_results[pair_key].append({
                        'Coefficient': coef,
                        'T-statistic': t_stat,
                        'P-value': p_val,
                        'Significant': p_val < 0.05
                    })
        
        for pair, results in test_results.items():
            test_results[pair] = pd.DataFrame(results)
            
        print("Performed statistical tests between languages.")
        return test_results
    
    def visualize_test_results(self, test_results):
        """
        Visualize statistical test results.
        """
        for pair, results_df in test_results.items():
            results_df['Comparison'] = pair
        
            plt.figure(figsize=(12, 6))
            pivot_df = pd.pivot_table(
                results_df,
                values='P-value',
                index='Coefficient',
                columns='Comparison'
            )
            sns.heatmap(
                pivot_df,
                annot=True,
                cmap='coolwarm_r',
                vmin=0,
                vmax=0.05
            )
            plt.title(f'P-values for {pair} (< 0.05 is significant)')
            plt.savefig(f'{pair}_pvalues.png')
            plt.close()
        
        summary_data = []
        for pair, results_df in test_results.items():
            significant_count = results_df['Significant'].sum()
            total_count = len(results_df)
            summary_data.append({
                'Comparison': pair,
                'Significant Coefficients': significant_count,
                'Total Coefficients': total_count,
                'Percentage': (significant_count / total_count) * 100
            })
            
        summary_df = pd.DataFrame(summary_data)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Comparison', y='Percentage', data=summary_df)
        plt.title('Percentage of Significantly Different MFCC Coefficients Between Languages')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)
        plt.savefig('significant_differences_summary.png')
        plt.close()
        print("Visualized statistical test results.")
        return summary_df
    
    def dimensional_analysis(self, processed_data):
        """
        Perform dimensional analysis using PCA.
        """
        all_features = []
        all_labels = []
        
        for sample in tqdm(processed_data, desc="Extracting features for PCA"):
            feature_vector = np.mean(sample['mfcc'], axis=1)
            all_features.append(feature_vector)
            all_labels.append(sample['language'])
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Language': y
        })
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='Language', data=pca_df, s=100, alpha=0.7)
        plt.title('PCA of MFCC Features by Language')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.savefig('pca_analysis.png')
        plt.close()
        print("Completed PCA analysis.")
        
        return pca_df, pca.explained_variance_ratio_
    
    def run_complete_analysis(self, max_files_per_language=10, num_visual_samples=3):
        """
        Run the complete analysis pipeline.
        """
        print("Starting complete audio analysis pipeline...")
        processed_data = self.process_audio_files(max_files_per_language=max_files_per_language)
        print("Visualizing MFCC spectrograms...")
        self.visualize_mfcc_spectrograms(processed_data, num_visual_samples)
        
        print("Computing MFCC statistics...")
        stats_df = self.compute_statistics(processed_data)
        self.visualize_statistics(stats_df)
        
        print("Performing statistical tests...")
        test_results = self.perform_statistical_tests(processed_data)
        summary_df = self.visualize_test_results(test_results)
        
        print("Performing dimensionality analysis...")
        pca_df, explained_variance = self.dimensional_analysis(processed_data)
        
        analysis_results = {
            'processed_data': processed_data,
            'statistics': stats_df,
            'statistical_tests': test_results,
            'test_summary': summary_df,
            'pca': pca_df,
            'pca_variance': explained_variance
        }
        
        print("Audio analysis pipeline completed.")
        return analysis_results


class IndianLanguageClassifier:
    def __init__(self, analyzer, test_size=0.2, random_state=42):
        """
        Initialize the classifier.
        """
        self.analyzer = analyzer
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.label_encoder = LabelEncoder()
        
    def split_data(self):
        """
        Split file paths into training and test sets.
        """
        all_files = []
        all_labels = []
        
        for language, file_list in self.analyzer.file_paths.items():
            all_files.extend(file_list)
            all_labels.extend([language] * len(file_list))
        
        train_files, test_files, train_labels, test_labels = train_test_split(
            all_files, all_labels, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=all_labels
        )
        
        print(f"Train set: {len(train_files)} files")
        print(f"Test set: {len(test_files)} files")
        return train_files, test_files, train_labels, test_labels
    
    def extract_features(self, processed_data):
        """
        Extract features from processed audio data.
        """
        features = []
        labels = []
        
        for item in tqdm(processed_data, desc="Extracting features"):
            mfcc_mean = np.mean(item['mfcc'], axis=1)
            mfcc_std = np.std(item['mfcc'], axis=1)
            delta_mean = np.mean(item['delta_mfcc'], axis=1)
            delta_std = np.std(item['delta_mfcc'], axis=1)
            delta2_mean = np.mean(item['delta2_mfcc'], axis=1)
            delta2_std = np.std(item['delta2_mfcc'], axis=1)
            
            feature_vector = np.concatenate([
                mfcc_mean, mfcc_std, 
                delta_mean, delta_std,
                delta2_mean, delta2_std
            ])
            
            features.append(feature_vector)
            labels.append(item['language'])
        
        X = np.array(features)
        y = self.label_encoder.fit_transform(labels)
        
        return X, y
    
    def prepare_data(self):
        """
        Prepare data for training.
        """
        train_files, test_files, _, _ = self.split_data()
        print("Processing training files...")
        train_data = self.analyzer.process_audio_files(file_paths=train_files)
        print("Processing test files...")
        test_data = self.analyzer.process_audio_files(file_paths=test_files)
        
        self.X_train, self.y_train = self.extract_features(train_data)
        self.X_test, self.y_test = self.extract_features(test_data)
        
        print(f"Training features shape: {self.X_train.shape}")
        print(f"Test features shape: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _train_with_randomized_search(self, pipeline, param_dist, tuning_fraction=0.2, cv=3, n_iter=5):
        """
        Helper function to perform randomized search on a subsample for hyperparameter tuning,
        and then retrain on the full training set.
        """
        # Subsample the training data
        if tuning_fraction < 1.0:
            X_tune, _, y_tune, _ = train_test_split(
                self.X_train, self.y_train, 
                test_size=(1 - tuning_fraction), 
                random_state=self.random_state, 
                stratify=self.y_train
            )
        else:
            X_tune, y_tune = self.X_train, self.y_train
        
        search = RandomizedSearchCV(
            pipeline, 
            param_distributions=param_dist, 
            cv=cv, 
            scoring='accuracy', 
            n_jobs=-1,
            n_iter=n_iter,
            random_state=self.random_state,
            verbose=1
        )
        search.fit(X_tune, y_tune)
        best_params = search.best_params_
        print("Best parameters found:", best_params)
        
        # Retrain the pipeline on the full training set using the best parameters.
        pipeline.set_params(**best_params)
        pipeline.fit(self.X_train, self.y_train)
        return pipeline

    def train_svm(self, tuning_fraction=0.2):
        """
        Train a Support Vector Machine classifier using RandomizedSearchCV and subsampling.
        """
        print("Training SVM classifier...")
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('svm', SVC(probability=True))
        ])
        
        param_dist = {
            'svm__C': [1, 10],
            'svm__gamma': ['scale', 0.1],
            'svm__kernel': ['rbf']  # narrowed to one kernel type for simplicity
        }
        
        best_pipeline = self._train_with_randomized_search(
            svm_pipeline, param_dist, tuning_fraction=tuning_fraction, cv=3, n_iter=5
        )
        self.models['svm'] = best_pipeline
        return best_pipeline
    
    def train_random_forest(self, tuning_fraction=0.2):
        """
        Train a Random Forest classifier using RandomizedSearchCV and subsampling.
        """
        print("Training Random Forest classifier...")
        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=self.random_state))
        ])
        
        param_dist = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [None, 10, 20],
            'rf__min_samples_split': [2, 5]
        }
        
        best_pipeline = self._train_with_randomized_search(
            rf_pipeline, param_dist, tuning_fraction=tuning_fraction, cv=3, n_iter=5
        )
        self.models['random_forest'] = best_pipeline
        return best_pipeline
    
    def train_neural_network(self, tuning_fraction=0.2):
        """
        Train a Neural Network classifier using RandomizedSearchCV and subsampling.
        """
        print("Training Neural Network classifier...")
        nn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('nn', MLPClassifier(random_state=self.random_state, max_iter=1000))
        ])
        
        param_dist = {
            'nn__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'nn__activation': ['relu', 'tanh'],
            'nn__alpha': [0.0001, 0.001],
            'nn__learning_rate': ['constant', 'adaptive']
        }
        
        best_pipeline = self._train_with_randomized_search(
            nn_pipeline, param_dist, tuning_fraction=tuning_fraction, cv=3, n_iter=5
        )
        self.models['neural_network'] = best_pipeline
        return best_pipeline
    
    def evaluate_models(self):
        """
        Evaluate all trained models on the test set.
        """
        results = {}
        for name, model in self.models.items():
            print(f"Evaluating model: {name}...")
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(
                self.y_test, 
                y_pred, 
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            results[name] = {
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            print(f"\n{name.upper()} RESULTS:")
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_))
        
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = self.models[best_model_name]
        print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
        return results
        
    def visualize_results(self, results):
        """
        Visualize the evaluation results.
        """
        plt.figure(figsize=(10, 6))
        accuracies = [results[model]['accuracy'] for model in results]
        sns.barplot(x=list(results.keys()), y=accuracies)
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.savefig('model_accuracy_comparison.png')
        plt.close()
        
        for name, result in results.items():
            plt.figure(figsize=(8, 6))
            cm = result['confusion_matrix']
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_
            )
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{name}.png')
            plt.close()
        
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        report = results[best_model_name]['report']
        classes = self.label_encoder.classes_
        class_metrics = []
        for cls in classes:
            if cls in report:
                metrics = report[cls]
                class_metrics.append({
                    'Language': cls,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-score': metrics['f1-score']
                })
        metrics_df = pd.DataFrame(class_metrics)
        plt.figure(figsize=(12, 8))
        metrics_df_melt = pd.melt(
            metrics_df, 
            id_vars=['Language'], 
            value_vars=['Precision', 'Recall', 'F1-score'],
            var_name='Metric',
            value_name='Score'
        )
        sns.barplot(x='Language', y='Score', hue='Metric', data=metrics_df_melt)
        plt.title(f'Per-class Performance Metrics - {best_model_name}')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('per_class_metrics.png')
        plt.close()
        print("Visualized evaluation results.")
    
    def train_all_models(self):
        """
        Train all classifier models.
        """
        self.train_svm()
        self.train_random_forest()
        self.train_neural_network()
        return self.models

    def predict_language(self, audio_file_path):
        """
        Predict the language of a new audio file.
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train_all_models() and evaluate_models() first.")
        
        processed_data = self.analyzer.process_audio_files(file_paths=[audio_file_path])
        if not processed_data:
            raise ValueError(f"Could not process audio file: {audio_file_path}")
        
        item = processed_data[0]
        mfcc_mean = np.mean(item['mfcc'], axis=1)
        mfcc_std = np.std(item['mfcc'], axis=1)
        delta_mean = np.mean(item['delta_mfcc'], axis=1)
        delta_std = np.std(item['delta_mfcc'], axis=1)
        delta2_mean = np.mean(item['delta2_mfcc'], axis=1)
        delta2_std = np.std(item['delta2_mfcc'], axis=1)
        
        feature_vector = np.concatenate([
            mfcc_mean, mfcc_std, 
            delta_mean, delta_std,
            delta2_mean, delta2_std
        ])
        feature_vector = feature_vector.reshape(1, -1)
        
        predicted_index = self.best_model.predict(feature_vector)[0]
        predicted_language = self.label_encoder.inverse_transform([predicted_index])[0]
        
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(feature_vector)[0]
            confidence = probabilities[predicted_index]
        else:
            confidence = None
        
        return predicted_language, confidence

    def run_complete_pipeline(self, max_files_per_language=20):
        """
        Run the complete pipeline: prepare data, train models, evaluate, visualize.
        """
        print("Starting complete classification pipeline...")
        # Pre-cache all features from the available dataset
        self.analyzer.process_audio_files(max_files_per_language=max_files_per_language)
        self.prepare_data()
        self.train_all_models()
        evaluation_results = self.evaluate_models()
        self.visualize_results(evaluation_results)
        print("Classification pipeline completed.")
        return evaluation_results


if __name__ == "__main__":
    data_dir = "/kaggle/input/audio-dataset-with-10-indian-languages/Language Detection Dataset"
    languages = ["Hindi", "Tamil", "Bengali"]

    print("Initializing analyzer...")
    analyzer = IndianLanguageAudioAnalyzer(data_dir, languages=languages, n_mfcc=13)
    
    # Run the complete analysis pipeline for audio feature exploration
    print("Running complete audio analysis...")
    analysis_results = analyzer.run_complete_analysis(max_files_per_language=20, num_visual_samples=3)
    
    print("Initializing classifier...")
    classifier = IndianLanguageClassifier(analyzer, test_size=0.2, random_state=42)
    
    # Run the complete classification pipeline
    print("Running complete classification pipeline...")
    classification_results = classifier.run_complete_pipeline(max_files_per_language=20)
    
    best_model_name = max(classification_results, key=lambda x: classification_results[x]['accuracy'])
    print(f"\nBest model: {best_model_name}")
    print(f"Accuracy: {classification_results[best_model_name]['accuracy']:.4f}")
