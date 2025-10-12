"""
Architecture du modèle student LEAF
Optimisé pour RTX 4050 6GB
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Dict, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class LEAFStudent(nn.Module):
    """
    Modèle student pour distillation LEAF-style

    Architecture compacte qui préserve les capacités d'embedding
    du teacher tout en étant beaucoup plus petit et rapide.
    """

    def __init__(
        self,
        teacher_config: Union[str, AutoConfig],
        compression_ratio: float = 4.0,
        hidden_size_ratio: float = 0.5,
        num_layers: int = 6,
        num_attention_heads: int = 6,
        pooling_mode: str = "mean",
    ):
        """
        Args:
            teacher_config: Config du teacher ou nom du modèle
            compression_ratio: Ratio de compression global (4.0 = 13x plus petit pour 300M -> 23M)
            hidden_size_ratio: Ratio de la hidden size (0.5 = 50% de la taille teacher)
            num_layers: Nombre de couches transformer
            num_attention_heads: Nombre de têtes d'attention
            pooling_mode: Mode de pooling ('mean', 'cls', 'max')
        """
        super().__init__()

        # Charger config teacher
        if isinstance(teacher_config, str):
            teacher_config = AutoConfig.from_pretrained(teacher_config)

        self.teacher_hidden_size = teacher_config.hidden_size
        self.pooling_mode = pooling_mode

        # Configuration student (plus compacte)
        student_config = self._create_student_config(
            teacher_config,
            hidden_size_ratio,
            num_layers,
            num_attention_heads
        )

        # Backbone transformer
        logger.info(f"Creating student model with config: {student_config}")
        self.encoder = AutoModel.from_config(student_config)

        # Projection layer pour alignment avec teacher
        # Permet au student d'avoir une dimension différente tout en
        # s'alignant avec les embeddings du teacher
        self.alignment_projection = nn.Sequential(
            nn.Linear(student_config.hidden_size, student_config.hidden_size * 2),
            nn.GELU(),
            nn.LayerNorm(student_config.hidden_size * 2),
            nn.Dropout(0.1),
            nn.Linear(student_config.hidden_size * 2, self.teacher_hidden_size)
        )

        # Normalisation finale
        self.output_norm = nn.LayerNorm(self.teacher_hidden_size)

        # Config pour tokenizer
        self.config = student_config
        self.tokenizer = None

        # Statistiques
        self._log_model_stats()

    def _create_student_config(
        self,
        teacher_config: AutoConfig,
        hidden_size_ratio: float,
        num_layers: int,
        num_attention_heads: int
    ) -> AutoConfig:
        """Crée la config student basée sur le teacher"""

        # Copier la config teacher
        student_config = AutoConfig.from_pretrained(teacher_config.name_or_path)

        # Ajuster les dimensions
        student_config.hidden_size = int(teacher_config.hidden_size * hidden_size_ratio)
        student_config.num_hidden_layers = num_layers
        student_config.num_attention_heads = num_attention_heads

        # S'assurer que hidden_size est divisible par num_attention_heads
        if student_config.hidden_size % num_attention_heads != 0:
            student_config.hidden_size = (
                (student_config.hidden_size // num_attention_heads) * num_attention_heads
            )

        # Ajuster intermediate size (FFN)
        student_config.intermediate_size = int(student_config.hidden_size * 4)

        return student_config

    def _log_model_stats(self):
        """Log les statistiques du modèle"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Student model initialized:")
        logger.info(f"  Total parameters: {total_params / 1e6:.2f}M")
        logger.info(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
        logger.info(f"  Hidden size: {self.config.hidden_size}")
        logger.info(f"  Num layers: {self.config.num_hidden_layers}")
        logger.info(f"  Num heads: {self.config.num_attention_heads}")

    def _pool_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool les embeddings de tokens en un seul vecteur

        Args:
            token_embeddings: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            pooled_embeddings: [batch_size, hidden_size]
        """
        if self.pooling_mode == "mean":
            # Mean pooling avec masque
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

        elif self.pooling_mode == "cls":
            # Utiliser le token [CLS]
            return token_embeddings[:, 0]

        elif self.pooling_mode == "max":
            # Max pooling avec masque
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9
            return torch.max(token_embeddings, dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_all: Si True, retourne les embeddings intermédiaires

        Returns:
            Dict contenant:
                - embeddings: Embeddings alignés avec teacher [batch_size, teacher_hidden_size]
                - student_embeddings: Embeddings natifs student [batch_size, student_hidden_size]
        """
        # Encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Token embeddings
        token_embeddings = outputs.last_hidden_state

        # Pooling
        student_embeddings = self._pool_embeddings(token_embeddings, attention_mask)

        # Projection pour alignment avec teacher
        aligned_embeddings = self.alignment_projection(student_embeddings)
        aligned_embeddings = self.output_norm(aligned_embeddings)

        result = {
            'embeddings': aligned_embeddings,
            'student_embeddings': student_embeddings,
        }

        if return_all:
            result['token_embeddings'] = token_embeddings
            result['hidden_states'] = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None

        return result

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
        device: Optional[str] = None
    ) -> torch.Tensor:
        """
        Encode des textes en embeddings

        Args:
            texts: Texte(s) à encoder
            batch_size: Taille des batchs pour l'encodage
            show_progress: Afficher une barre de progression
            normalize: Normaliser les embeddings
            device: Device à utiliser (cuda/cpu)

        Returns:
            embeddings: [num_texts, embedding_dim]
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call set_tokenizer() first.")

        # S'assurer que c'est une liste
        if isinstance(texts, str):
            texts = [texts]

        # Device
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        self.eval()
        all_embeddings = []

        # Progress bar
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(device)

                # Forward
                outputs = self.forward(**inputs)
                embeddings = outputs['embeddings']

                # Normalize
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def set_tokenizer(self, tokenizer: AutoTokenizer):
        """Configure le tokenizer"""
        self.tokenizer = tokenizer

    def save_pretrained(self, save_path: str):
        """Sauvegarde le modèle"""
        import os
        os.makedirs(save_path, exist_ok=True)

        # Sauvegarder les poids
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'pooling_mode': self.pooling_mode,
        }, os.path.join(save_path, 'pytorch_model.bin'))

        # Sauvegarder le tokenizer si disponible
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = 'cpu'):
        """Charge le modèle depuis un checkpoint"""
        checkpoint = torch.load(
            os.path.join(load_path, 'pytorch_model.bin'),
            map_location=device
        )

        # Recréer le modèle
        # Note: Vous devrez adapter ceci selon votre config
        model = cls(
            teacher_config=checkpoint['config'],
            pooling_mode=checkpoint.get('pooling_mode', 'mean')
        )

        model.load_state_dict(checkpoint['model_state_dict'])

        # Charger le tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(load_path)
            model.set_tokenizer(tokenizer)
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")

        logger.info(f"Model loaded from {load_path}")
        return model


def create_student_from_teacher(
    teacher_name: str,
    compression_ratio: float = 4.0,
    device: str = 'cuda'
) -> LEAFStudent:
    """
    Crée un modèle student depuis un teacher

    Args:
        teacher_name: Nom du modèle teacher sur HuggingFace
        compression_ratio: Ratio de compression
        device: Device

    Returns:
        student: Modèle student initialisé
    """
    logger.info(f"Creating student from teacher: {teacher_name}")

    # Charger config teacher
    teacher_config = AutoConfig.from_pretrained(teacher_name)

    # Créer student
    student = LEAFStudent(
        teacher_config=teacher_config,
        compression_ratio=compression_ratio
    ).to(device)

    # Charger tokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    student.set_tokenizer(tokenizer)

    return student
