import os
import sys
# from flask_cors import CORS

# Set NLTK data path for Azure deployment
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if os.path.exists(nltk_data_path):
    import nltk
    nltk.data.path.append(nltk_data_path)

from flask import Flask, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import math
import numpy as np
from scipy.stats import entropy, norm
import sympy as sp
import torch
import torch.nn as nn
import logging

# Initialize Flask app
app = Flask(__name__)

# CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data on startup if not already present
def ensure_nltk_data():
    try:
        # Try to access the data first
        word_tokenize("test")
        stopwords.words('english')
        logger.info("NLTK data already available")
    except Exception as e:
        logger.info(f"Downloading NLTK data: {e}")
        try:
            nltk.download(['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger'], quiet=True)
        except Exception as download_error:
            logger.warning(f"NLTK download warning: {download_error}")

ensure_nltk_data()

class ValuePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x):
        return self.fc(x)

def train_simple_nn(features, targets):
    model = ValuePredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(100):
        pred = model(features)
        loss = loss_fn(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def fetch_market_avg(project_type):
    market_data = {"recipe": 15.0, "software code": 500.0, "business plan": 200.0}
    return market_data.get(project_type.lower(), 100.0)

def calculate_flesch_kincaid(text):
    try:
        sentences = len(nltk.sent_tokenize(text))
        words = len(word_tokenize(text))
        syllables = sum(len(word) // 3 for word in word_tokenize(text))
        if sentences == 0 or words == 0:
            return 0
        return 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    except Exception:
        return 0

def compute_uniqueness_score(text):
    try:
        words = [w.lower() for w in word_tokenize(text) if w.isalpha() and w not in stopwords.words('english')]
        if not words:
            return 0
        freq_dist = nltk.FreqDist(words)
        probs = [freq / len(words) for freq in freq_dist.values()]
        return entropy(probs) / math.log(len(set(words)) + 1)
    except Exception:
        return 0

def advanced_tokenization_algorithm(content, project_type, development_cost, stage_completion,
                                     expected_revenue, discount_rate, risk_factor,
                                     user_base, token_supply, useful_life, blockchain_flag):
    # Validation
    if not content or development_cost < 0 or stage_completion < 0 or stage_completion > 100:
        raise ValueError("Invalid inputs.")

    word_count = len(word_tokenize(content))
    complexity_score = min(max(calculate_flesch_kincaid(content) / 20, 0), 1)
    uniqueness_score = compute_uniqueness_score(content)
    length_score = min(word_count / 1000, 1)
    intrinsic_score = (complexity_score * 0.4 + uniqueness_score * 0.4 + length_score * 0.2)

    # EVM calculations
    BAC = development_cost / (stage_completion / 100) if stage_completion > 0 else development_cost
    EV = (stage_completion / 100) * BAC
    AC = development_cost
    CPI = EV / AC if AC > 0 else 1
    eac_sym = sp.symbols('EAC')
    eqn = sp.Eq(eac_sym, AC + (BAC - EV) / CPI)
    EAC = float(sp.solve(eqn, eac_sym)[0])

    # NPV calculation
    growth_rate = 0.05
    cash_flows = [expected_revenue * (1 + growth_rate)**t for t in range(1, 6)]
    NPV = sum(cf / (1 + discount_rate)**t for t, cf in enumerate(cash_flows, 1)) - development_cost

    # Amortization
    amortized_value = development_cost * (1 - (1 / useful_life))

    # Tokenomics
    tokenomics_adjust = 0
    if blockchain_flag:
        network_value = 0.001 * user_base**2
        if token_supply:
            token_price = network_value / token_supply
            tokenomics_adjust = token_price * token_supply
        else:
            tokenomics_adjust = network_value

    market_avg = fetch_market_avg(project_type)

    # Monte Carlo simulation
    sims = 1000
    npv_sims = []
    for _ in range(sims):
        sim_revenue = norm.rvs(expected_revenue, expected_revenue * risk_factor)
        sim_cf = [sim_revenue * (1 + growth_rate)**t for t in range(1, 6)]
        sim_NPV = sum(cf / (1 + discount_rate)**t for t, cf in enumerate(sim_cf, 1)) - development_cost * norm.rvs(1, risk_factor)
        npv_sims.append(sim_NPV)
    mean_value = np.mean(npv_sims)
    low, high = np.percentile(npv_sims, [5, 95])

    # ML prediction
    features = torch.tensor([[intrinsic_score, CPI, NPV, amortized_value, tokenomics_adjust]] * 100, dtype=torch.float32)
    targets = torch.tensor([[mean_value]] * 100, dtype=torch.float32).unsqueeze(1)
    model = train_simple_nn(features, targets)
    pred_input = torch.tensor([[intrinsic_score, CPI, NPV, amortized_value, tokenomics_adjust]], dtype=torch.float32)
    ml_adjusted_value = model(pred_input).item()

    # Final calculation
    base_value = (market_avg * 0.2) + (EAC * 0.2) + (NPV * 0.2) + (amortized_value * 0.1) + (tokenomics_adjust * 0.1) + (ml_adjusted_value * 0.2)
    estimated_value = base_value * (stage_completion / 100) * (1 + intrinsic_score)

    # Return structured data instead of formatted string
    return {
        "intrinsic_score": round(intrinsic_score, 2),
        "evm": {
            "ev": round(EV, 2),
            "cpi": round(CPI, 2),
            "eac": round(EAC, 2)
        },
        "economics": {
            "npv": round(NPV, 2)
        },
        "accounting": {
            "amortized_value": round(amortized_value, 2)
        },
        "tokenomics": round(tokenomics_adjust, 2),
        "probabilistic": {
            "mean": round(mean_value, 2),
            "confidence_interval_90": {
                "low": round(low, 2),
                "high": round(high, 2)
            }
        },
        "ml_adjusted": round(ml_adjusted_value, 2),
        "final_estimated_value": round(estimated_value, 2)
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Tokenization API is running"})

@app.route('/tokenize', methods=['POST'])
def tokenize():
    """Main tokenization endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract parameters with defaults
        content = data.get('content', '')
        project_type = data.get('project_type', 'recipe')
        development_cost = float(data.get('development_cost', 0.0))
        stage_completion = float(data.get('stage_completion', 100.0))
        expected_revenue = float(data.get('expected_revenue', 0.0))
        discount_rate = float(data.get('discount_rate', 0.1))
        risk_factor = float(data.get('risk_factor', 0.2))
        user_base = int(data.get('user_base', 0))
        token_supply = data.get('token_supply')
        if token_supply is not None and token_supply > 0:
            token_supply = int(token_supply)
        else:
            token_supply = None
        useful_life = int(data.get('useful_life', 5))
        blockchain_flag = bool(data.get('blockchain_flag', False))
        
        # Call the algorithm
        result = advanced_tokenization_algorithm(
            content, project_type, development_cost, stage_completion,
            expected_revenue, discount_rate, risk_factor,
            user_base, token_supply, useful_life, blockchain_flag
        )
        
        return jsonify({
            "success": True,
            "data": result
        })
        
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/tokenize/example', methods=['GET'])
def example_request():
    """Returns an example request format"""
    example = {
        "content": "This is a sample recipe for chocolate cake with unique ingredients and complex preparation steps.",
        "project_type": "recipe",
        "development_cost": 1000.0,
        "stage_completion": 85.0,
        "expected_revenue": 5000.0,
        "discount_rate": 0.1,
        "risk_factor": 0.2,
        "user_base": 1000,
        "token_supply": 10000,
        "useful_life": 5,
        "blockchain_flag": True
    }
    
    return jsonify({
        "message": "Example request format for POST /tokenize",
        "example_request": example
    })

if __name__ == '__main__':
    # For development only
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)