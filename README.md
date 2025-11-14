A multimodal emotion recognition system that combines Generative AI, Responsible AI, and Affective Computing to analyze human emotions from video inputs. The system processes three modalities—
-text (DistilRoBERTa), 
-audio (fine-tuned Wav2Vec2)
-facial expressions (DeepFace)

Afterwards, fuses their predictions using majority voting to achieve 81% accuracy across 7 emotion classes (Happy, Sad, Angry, Disgust, Fear, Surprise, Neutral).
Built with ethical AI safeguards including bias detection (Detoxify) and threat filtering, the pipeline generates human-readable emotional summaries using FLAN-T5. Trained on CREMA-D dataset, the system demonstrates robust performance in real-time emotion detection with applications in mental health monitoring, interactive storytelling and educational platforms.


