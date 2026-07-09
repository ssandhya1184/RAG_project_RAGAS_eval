import os
from dotenv import load_dotenv
load_dotenv()

PDF_PATH = "data/optima-secure-brochure.pdf"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "BAAI/bge-small-en"

LLM_MODEL = "gemma-4-31b-it"
JUDGE_LLM = "gemini-2.5-flash"
PROVIDER="google"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

CHUNKING_STRATEGY = "fixed" # options: fixed/semantic/hierarchical

RETRIEVAL_STRATEGY = "similarity" # options: similarity/ mmr / hybrid

TOP_K = 3
MMR_K = 5
#MMR Score = λ × relevance - (1 - λ) × max_similarity_to_selected
MMR_LAMBDA = 0.5


ENABLE_RERANKING = True
RERANK_TOP_K = 3

RERANK_MODEL = "BAAI/bge-reranker-base"

QWEN_MODEL="qwen/qwen3.6-27b"

QUESTION_LIST = ["Are maternity expenses covered under this policy?",
                      "Is there any co-payment based on geography?",
                       "Can the policy be renewed for life?",
                       "What are the standard exclusions in this policy?",
                       "What is Optima Secure?",
                        "How much coverage does Optima Secure claim to provide overall?",
                        "What is the Secure Benefit in this policy?",
                        "What does the Plus Benefit do?",
                        "What is the Restore Benefit?",
                        "What expenses are covered under the Protect Benefit?",
                        "How many network hospitals are available under this policy?",
                        "What is the pre and post hospitalization coverage duration?",
                        "Does the policy cover home healthcare?",
                        "What is the daily cash benefit for shared accommodation?",
                        "How does Optima Secure achieve 4X coverage using different benefits?",
                        "If a person buys a 10 lakh base cover, how does it grow after 2 years?",
                        "How do Secure Benefit and Plus Benefit work together?",
                        "What happens to coverage if a claim is made during the policy year?",
                        "How does Restore Benefit impact total available coverage after a claim?",
                        "How does the deductible affect the premium and coverage?",
                        "If Mr. Sharma pays ₹26,700 premium, how much does he pay after opting for deductible?",
                        "What is the percentage increase in coverage after 1 year and 2 years?",		
                        "What is the maximum daily cash benefit and total limit?",
                        "How much discount can be obtained by choosing higher deductibles?",
                        "What is the maximum air ambulance coverage?",
                        "If a policyholder makes multiple claims in a year, how does the coverage get restored?",
                        "How would the policy behave for someone choosing a family floater vs individual plan?",
                        "What happens if a user opts for Unlimited Restore add-on?",
                        "If hospitalization includes non-medical expenses, how are they handled?",		
                        "Does the policy reduce benefits if a claim is made?",
                        "Are maternity expenses covered under this policy?",
                        "Is there any co-payment based on geography?",
                        "Can the policy be renewed for life?",
                        "What are the standard exclusions in this policy?"
                       ]

GROUND_TRUTH = ["The brochure pages provided do not mention maternity coverage. Therefore, this information cannot be confirmed from the brochure alone.",
            "The brochure does not mention any geography-based co-payment provisions. Therefore, this information cannot be confirmed from the brochure alone.",
            "The brochure pages provided do not explicitly mention lifelong renewability. Therefore, this information cannot be confirmed from the brochure alone." ,
            "The brochure pages provided do not include the policy exclusions section. Therefore, the standard exclusions cannot be determined from the brochure content provided." ,
            "It covers non-medical expenses like gloves, masks, and consumables during hospitalization." ,       
            "Optima Secure claims to provide up to 4X coverage of the chosen base cover through Secure Benefit, Plus Benefit, Restore Benefit, and Protect Benefit.",
            "Secure Benefit instantly and automatically doubles the insurance cover (2X coverage) from Day 1 at no additional cost." ,
            "Plus Benefit automatically increases the base cover by 50%% after 1 year and 100%% after 2 years, irrespective of claims made." ,
            "Restore Benefit automatically restores 100%% of the base cover if any partial or full claim is made during the policy year, at no additional cost." ,
            "Protect Benefit covers listed non-medical consumables and expenses incurred during hospitalization, including items such as gloves, masks, nebulizer kits, and other eligible consumables."  ,
            "The policy provides access to 16,000+ network hospitals and healthcare service providers." ,
            "Optima Secure covers medical expenses for 60 days before hospitalization and 180 days after hospitalization."  ,
            "Yes. The policy covers home healthcare treatment on a cashless basis, including doctor visits, nursing charges, and other eligible medical expenses when hospitalization would normally have been required."  ,
            "The policy provides ₹800 per day, up to a maximum of ₹4,800, when the insured opts for a shared room in a network hospital and the hospitalization exceeds 48 hours." ,
            "Optima Secure achieves up to 4X coverage by combining four built-in benefits: Secure Benefit doubles the base cover from Day 1, Plus Benefit increases the base cover by 50%% after 1 year and 100%% after 2 years, Restore Benefit restores 100%% of the base cover after a claim, and Protect Benefit covers eligible non-medical expenses during hospitalization. Together these benefits can increase the effective coverage to four times the original base cover."  ,
            "A ₹10 lakh base cover becomes ₹20 lakh immediately through Secure Benefit. After 2 years, Plus Benefit increases the base cover to ₹20 lakh. Combined with Secure Benefit, total available coverage becomes ₹40 lakh."  ,
            "Secure Benefit immediately doubles the base cover, while Plus Benefit gradually increases the base cover by 50%% after 1 year and 100%% after 2 years. Together they increase the total available coverage significantly over time." ,
            "If a claim is made during the policy year, Restore Benefit automatically restores 100%% of the base cover, ensuring continued coverage availability."  ,
            "Restore Benefit replenishes the entire base sum insured after a partial or full claim, allowing the policyholder to continue using coverage for subsequent claims during the same policy year."  ,
            "A deductible requires the policyholder to pay an agreed amount of claim expenses before insurance coverage applies. Choosing a deductible reduces the premium, with discounts ranging from 15% to 65%% depending on the deductible amount and base sum insured."  ,
            "By choosing a deductible of ₹25,000, Mr. Sharma receives a 25% premium discount, reducing his premium from ₹26,700 to ₹20,025."  ,
            "The Plus Benefit increases the base cover by 50%% after 1 year and by 100%% after 2 years."      ,
            "The policy provides ₹800 per day for shared accommodation, subject to a maximum limit of ₹4,800 per hospitalization."  ,
            "Depending on the deductible amount selected, premium discounts can go up to 65%."  ,
            "The brochure states that emergency air ambulance services are covered but does not specify a maximum coverage amount.",
            "After a claim is made, Restore Benefit automatically restores 100%% of the base cover. Additionally, policyholders may opt for the Unlimited Restore add-on, which provides unlimited restorations during a policy year"  ,
            "In an Individual policy, each insured member has a separate sum insured, and up to 6 adults and 6 children can be covered. In a Family Floater policy, all covered members share a common sum insured, with a maximum of 4 adults and 6 children covered."  ,
            "The Unlimited Restore add-on provides unlimited restorations of eligible coverage during the policy year, allowing repeated replenishment after claims." ,
            "Eligible non-medical consumables and expenses such as gloves, masks, and nebulizer kits are covered under the Protect Benefit with no deduction." ,
            "No. The brochure explicitly states that Plus Benefit increases coverage irrespective of claims made, and Restore Benefit replenishes the base cover after claims.",
            "The brochure pages provided do not mention maternity coverage. Therefore, this information cannot be confirmed from the brochure alone."  ,
            "The brochure does not mention any geography-based co-payment provisions. Therefore, this information cannot be confirmed from the brochure alone."  ,
            "The brochure pages provided do not explicitly mention lifelong renewability. Therefore, this information cannot be confirmed from the brochure alone." ,
            "The brochure pages provided do not include the policy exclusions section. Therefore, the standard exclusions cannot be determined from the brochure content provided."  
]

QUESTION_LIST1 = [
       "Can the policy be renewed for life?",
       
       "What is the Restore Benefit?",
     
       "How many network hospitals are available under this policy?"

       ]

GROUND_TRUTH1 = ["The brochure pages provided do not explicitly mention lifelong renewability. Therefore, this information cannot be confirmed from the brochure alone." ,
            "Restore Benefit automatically restores 100%% of the base cover if any partial or full claim is made during the policy year, at no additional cost." ,
           
            "The policy provides access to 16,000+ network hospitals and healthcare service providers." ,
           
     ]
        

