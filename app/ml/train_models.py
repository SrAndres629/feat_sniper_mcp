He analizado `app/ml/train_models.py` y confirmado los requisitos basándome en `nexus_brain/hybrid_model.py`.

He detectado que el entorno actual no me permite modificar archivos directamente (las herramientas de edición no están disponibles). Como Gemini CLI, te solicito que apliques la siguiente refactorización a la función `train_lstm` en `app/ml/train_models.py`.

**Cambios requeridos:**
1.  Cambiar la clave de guardado de `"model_state"` a `"state_dict"`.
2.  Convertir los arrays de numpy en `scaler_stats` a listas de Python (`.tolist()`) para asegurar la compatibilidad con JSON.

Aquí tienes el código corregido para la función `train_lstm`. Por favor, sustituye la función existente con esta versión:

def train_lstm(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN) -> str:
    """
    Entrena LSTM con Attention para capturar patrones secuenciales.
    CORRECCION AUDITORIA: Scaling Causal (Fit en el 80% inicial).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    
    logger.info("=" * 60)
    logger.info("PHASE 3: Training LSTM with Attention (Causal Scaling)")
    logger.info("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 1. Causal Normalization Strategy
    # We must fit scaler ONLY on the training portion of the data
    # LSTM typically uses a time-split (e.g., first 80% train, last 20% val)
    split_idx = int(len(X) * 0.8)
    
    X_train_raw = X[:split_idx]
    
    scaler = StandardScaler()
    scaler.fit(X_train_raw) # Fit only on past
    
    X_scaled = scaler.transform(X) # Apply to all (future uses past stats)
    
    logger.info(f"Scaler fitted on first {split_idx} samples.")
    
    # Crear secuencias
    def make_sequences(X_in: np.ndarray, y_in: np.ndarray, seq_len: int):
        """Convierte datos tabulares a secuencias 3D."""
        seqs, labels = [], []
        for i in range(seq_len, len(X_in)):
            seqs.append(X_in[i - seq_len:i, :])
            labels.append(y_in[i])
        return (
            torch.tensor(np.stack(seqs), dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
    
    X_seq, y_seq = make_sequences(X_scaled, y, seq_len)
    logger.info(f"Created {len(X_seq)} sequences of length {seq_len}")
    
    # Split temporal (80/20) - Must match the scaling split logic roughly
    # Actually, X_seq is slightly smaller than X, but the boundary is consistent 
    # because X_scaled preserves order.
    train_size = int(0.8 * len(X_seq))
    
    X_train, X_val = X_seq[:train_size], X_seq[train_size:]
    y_train, y_val = y_seq[:train_size], y_seq[train_size:]
    
    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=False  # No shuffle for time series
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=64,
        shuffle=False
    )
    
    # Model init
    input_dim = X.shape[1]
    model = LSTMWithAttention(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        num_classes=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    best_model_state = None
    
    # Training Loop
    logger.info("Starting LSTM training...")
    for epoch in range(20): # 20 Epochs
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)
                
        val_acc = val_correct / total_val if total_val > 0 else 0
        logger.info(f"Epoch {epoch+1}: Loss={(train_loss/len(train_loader)):.4f}, ValAcc={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()
            
    # Save Model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"lstm_{SYMBOL}_v2.pt")
    
    # Save scaler statistics for inference!
    scaler_stats = {
        "mean": scaler.mean_.tolist(), # FIXED: Convert to list
        "scale": scaler.scale_.tolist() # FIXED: Convert to list
    }
    
    torch.save({
        "state_dict": best_model_state, # FIXED: Key name
        "scaler_stats": scaler_stats, # Persist Causal Scaler
        "model_config": {
            "input_dim": input_dim,
            "hidden_dim": 64,
            "num_layers": 2,
            "seq_len": seq_len
        },
        "trained_at": datetime.utcnow().isoformat(),
        "best_acc": best_acc
    }, model_path)
    
    logger.info(f" LSTM Model saved to {model_path} (Acc: {best_acc:.4f})")
    return model_path

Por favor confirma cuando hayas realizado el cambio para que pueda verificarlo.