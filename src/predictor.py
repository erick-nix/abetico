"""
Módulo responsável por fazer predições de diabetes usando modelo treinado
"""
import numpy as np
import joblib
import os


class DiabetesPredictor:
    """Classe para fazer predições de diabetes"""
    
    def __init__(self):
        """Inicializa o preditor e carrega o modelo"""
        self.model = None
        self.scaler = None
        self.features = ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        self.load_model()
    
    def load_model(self):
        """Carrega o modelo treinado e o scaler"""
        # Caminho para os arquivos do modelo
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, 'src', 'model', 'diabetes_model.pkl')
        scaler_path = os.path.join(base_dir, 'src', 'model', 'scaler.pkl')
        
        try:
            # Carregar modelo
            self.model = joblib.load(model_path)
            print(f"Modelo carregado: {model_path}")
            
            # Carregar scaler
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler carregado: {scaler_path}")
        except FileNotFoundError as e:
            print(f"Erro: Modelo não encontrado em {model_path}")
            print("   Execute: python notebooks/train_model.py")
            print("   Depois mova os arquivos .pkl para src/model/")
            raise e
    
    def predict(self, glucose, imc, historico_familiar, idade):
        """
        Faz a predição de diabetes usando o modelo treinado
        
        Args:
            glucose (float): Nível de glicose no sangue (mg/dL)
            imc (float): Índice de Massa Corporal (kg/m²)
            historico_familiar (float): DiabetesPedigreeFunction (0.0 a 2.5)
            idade (int): Idade do paciente em anos
        
        Returns:
            float: Probabilidade de ter diabetes (0 a 1)
        """
        if self.model is None or self.scaler is None:
            print("Modelo não carregado! Usando predição simulada...")
            return self._fallback_prediction(glucose, imc, historico_familiar, idade)
        
        try:
            # Preparar dados na ordem correta das features
            # ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            X = np.array([[glucose, imc, historico_familiar, idade]])
            
            # Normalizar com o scaler
            X_scaled = self.scaler.transform(X)
            
            # Fazer predição (retorna probabilidade da classe 1 = diabético)
            probability = self.model.predict_proba(X_scaled)[0][1]
            
            return float(probability)
            
        except Exception as e:
            print(f"Erro na predição: {e}")
            return self._fallback_prediction(glucose, imc, historico_familiar, idade)
    
    def _fallback_prediction(self, glucose, imc, historico_familiar, idade):
        """
        Predição de fallback caso o modelo não esteja disponível
        Usa regras médicas simples
        """
        risk_score = 0.0
        
        # Glicose (peso: 40%)
        if glucose >= 126:  # Diabetes
            risk_score += 0.40
        elif glucose >= 100:  # Pré-diabetes
            risk_score += 0.25
        else:
            risk_score += 0.10
        
        # IMC (peso: 30%)
        if imc >= 30:  # Obesidade
            risk_score += 0.30
        elif imc >= 25:  # Sobrepeso
            risk_score += 0.20
        else:
            risk_score += 0.05
        
        # Histórico familiar (peso: 20%)
        risk_score += min(historico_familiar / 2.5, 1.0) * 0.20
        
        # Idade (peso: 10%)
        if idade >= 45:
            risk_score += 0.10
        elif idade >= 35:
            risk_score += 0.05
        
        return min(max(risk_score, 0.0), 1.0)

    def get_risk_description(self, probability):
        """
        Retorna a descrição baseado na probabilidade
        
        Args:
            probability (float): Probabilidade de ter diabetes (0 a 1)
        
        Returns:
            str: Descrição do nível de risco
        """
        if probability < 0.3:
            return "Risco baixo de diabetes"
        elif probability < 0.6:
            return "Risco moderado de diabetes. Considere consultar um médico."
        else:
            return "Risco alto de diabetes. Recomenda-se consultar um médico urgentemente."