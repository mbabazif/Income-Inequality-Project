from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException
import uvicorn

# Create a FastAPI instance
app = FastAPI()

# Load the entire pipeline
rfc_pipeline = joblib.load('./rfc_pipeline.joblib')
encoder = joblib.load('./encoder.joblib')

# Define a FastAPI instance ML model input schema
class IncomePredictionInput(BaseModel):
    age:                   int
    gender:                object 
    education:             object 
    worker_class:          object 
    marital_status:        object 
    race:                  object 
    is_hispanic:           object 
    employment_commitment: object 
    employment_stat:       int  
    wage_per_hour:         int  
    working_week_per_year: int  
    industry_code:         int  
    industry_code_main:    object 
    occupation_code:       int  
    occupation_code_main:  object 
    total_employed:        int 
    household_summary:     object 
    vet_benefit:           int 
    tax_status:            object 
    gains:                 int 
    losses:                int  
    stocks_status:         int  
    citizenship:           object 
    importance_of_record:  float
 
class IncomePredictionOutput(BaseModel):
    income_prediction: str
    prediction_probability: float   
# Defining the root endpoint for the API
@app.get("/")
def index():
    explanation = {
        'message': "Welcome to the Income Iniquality Prediction App",
        'description': "This API allows you to predict Income Iniquality based on Personal data.",
    }
    return explanation


@app.post('/classify', response_model=IncomePredictionOutput)
def income_classification(income: IncomePredictionInput):
    try:
        df = pd.DataFrame([income.model_dump()])
           
        # Make predictions
        prediction = rfc_pipeline.predict(df)
        output = rfc_pipeline.predict_proba(df)

        prediction_result = "Income over $50K" if prediction[0] == 1 else "Income under $50K"
        return {"income_prediction": prediction_result, "prediction_probability": output[0][1]}


    except Exception as e:
        # Return error message and details if an exception occurs
        error_detail = str(e)
        raise HTTPException(status_code=500, detail=f"Error during classification: {error_detail}")


if __name__ == '__main__':
    uvicorn.run('main:app', reload=True)