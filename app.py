from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd


model = joblib.load("xgboost_model.pkl")


TRANSACTION_TYPES = {
    'TRANSFER': 0,
    'CASH_OUT': 1,
    'DEBIT': 2,
    'CASH_IN': 3
}


app = Flask(__name__)

def generate_missing_values(transaction_type, amount):
    oldbalanceDest = 0.0
    newbalanceDest = 0.0
    
    if transaction_type == 0:  # TRANSFER
        oldbalanceDest = 0.0
        newbalanceDest = 0.0  # On ne connaît pas le solde du destinataire
    elif transaction_type == 1:  # CASH_OUT
        oldbalanceDest = 0.0
        newbalanceDest = 0.0  # Retrait sans impact sur un autre compte connu
    elif transaction_type == 3:  # CASH_IN
        oldbalanceDest = 0.0  # On suppose un compte vide
        newbalanceDest = amount  # Le destinataire reçoit le montant complet
    
    return oldbalanceDest, newbalanceDest

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Recuperer les donnees du formulaire
        data = request.form.to_dict()
        
        # Convertir les valeurs en float sauf pour type_transaction
        transaction_type = TRANSACTION_TYPES[data["type_transaction"]]
        amount = float(data["amount"])
        oldbalanceOrg = float(data["oldbalanceOrg"])
        newbalanceOrig = float(data["newbalanceOrig"])
        
        # Generer oldbalanceDest et newbalanceDest 
        oldbalanceDest, newbalanceDest = generate_missing_values(transaction_type, amount)
        
        # Creer un DataFrame pour correspondre au modèle
        step = 1  # Valeur par défaut
        features = [step, transaction_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
        df_features = pd.DataFrame([features], columns=['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])
        
        
        prediction = model.predict(df_features)[0]
        
        
        result = "🚨 Fraude détectée ! 🚨" if prediction == 1 else "✅ Transaction normale."
        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
