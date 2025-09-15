from mini import fetch_question_data

tf_df = fetch_question_data(
    question_id=6231,
    metabase_url="https://metabase.ofood.cloud",
    username="a.mehmandoost@OFOOD.CLOUD",
    password="Fff322666@",
    team="data",
    workers=8,
    page_size=50000,
)