













(regtriage-openenv) hermes@ree2raz:~/projects/RegTriage-OpenEnv$ uv run python inference.py
[STEP] step=1 action=get_call_metadata() reward=0.05 done=false error=null
[STEP] step=2 action=get_sentiment_timeline() reward=0.05 done=false error=null
[STEP] step=3 action=read_transcript_chunk(start_turn=0,end_turn=2) reward=0.02 done=false error=null
[STEP] step=4 action=flag_violation(violation_type=regulatory_disclosure_failure,violation_severity=high) reward=0.00 done=false error=null
[STEP] step=5 action=read_transcript_chunk(start_turn=3,end_turn=5) reward=0.02 done=false error=null
[STEP] step=6 action=read_transcript_chunk(start_turn=6,end_turn=8) reward=-0.02 done=false error=Invalid range [6, 8]. Valid: [0, 7].
[STEP] step=7 action=read_transcript_chunk(start_turn=6,end_turn=7) reward=0.02 done=false error=null
[STEP] step=8 action=submit_report(compliance_pass=False) reward=0.90 done=true error=null
[END] success=true steps=8 score=0.898 rewards=0.05,0.05,0.02,0.00,0.02,-0.02,0.02,0.90
[START] task=call_002 env=regtriage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=get_call_metadata() reward=0.05 done=false error=null
[STEP] step=2 action=get_sentiment_timeline() reward=0.05 done=false error=null
[STEP] step=3 action=read_transcript_chunk(start_turn=0,end_turn=2) reward=0.02 done=false error=null
[STEP] step=4 action=read_transcript_chunk(start_turn=7,end_turn=9) reward=0.02 done=false error=null
[STEP] step=5 action=analyze_turn(turn_index=2,policy_hypothesis=pii_exposure_risk) reward=0.02 done=false error=null
[STEP] step=6 action=read_transcript_chunk(start_turn=3,end_turn=6) reward=0.02 done=false error=null
[STEP] step=7 action=submit_report(compliance_pass=True) reward=0.88 done=true error=null
[END] success=true steps=7 score=0.880 rewards=0.05,0.05,0.02,0.02,0.02,0.02,0.88
[START] task=call_003 env=regtriage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=get_call_metadata() reward=0.05 done=false error=null
[STEP] step=2 action=get_sentiment_timeline() reward=0.05 done=false error=null
[STEP] step=3 action=read_transcript_chunk(start_turn=0,end_turn=2) reward=0.02 done=false error=null
[STEP] step=4 action=read_transcript_chunk(start_turn=5,end_turn=7) reward=0.02 done=false error=null
[STEP] step=5 action=flag_violation(violation_type=pii_exposure_risk,violation_severity=high,turn_index=5) reward=0.00 done=false error=nul
[STEP] step=6 action=read_transcript_chunk(start_turn=8,end_turn=12) reward=0.02 done=false error=null
[STEP] step=7 action=submit_report(compliance_pass=False) reward=0.92 done=true error=null
[END] success=true steps=7 score=0.919 rewards=0.05,0.05,0.02,0.02,0.00,0.02,0.92
[START] task=call_004 env=regtriage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=get_call_metadata() reward=0.05 done=false error=null
[STEP] step=2 action=get_sentiment_timeline() reward=0.05 done=false error=null
[STEP] step=3 action=read_transcript_chunk(start_turn=0,end_turn=2) reward=0.02 done=false error=null
[STEP] step=4 action=read_transcript_chunk(start_turn=5,end_turn=7) reward=0.02 done=false error=null
[STEP] step=5 action=read_transcript_chunk(start_turn=2,end_turn=4) reward=0.02 done=false error=null
[STEP] step=6 action=read_transcript_chunk(start_turn=7,end_turn=9) reward=-0.02 done=false error=Invalid range [7, 9]. Valid: [0, 8].
[STEP] step=7 action=read_transcript_chunk(start_turn=7,end_turn=8) reward=0.02 done=false error=null
[STEP] step=8 action=submit_report(compliance_pass=True) reward=0.86 done=true error=null
[END] success=true steps=8 score=0.865 rewards=0.05,0.05,0.02,0.02,0.02,-0.02,0.02,0.86
[START] task=call_005 env=regtriage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=get_call_metadata() reward=0.05 done=false error=null
[STEP] step=2 action=get_sentiment_timeline() reward=0.05 done=false error=null
[STEP] step=3 action=read_transcript_chunk(start_turn=0,end_turn=2) reward=0.02 done=false error=null
[STEP] step=4 action=read_transcript_chunk(start_turn=6,end_turn=8) reward=0.02 done=false error=null
[STEP] step=5 action=read_transcript_chunk(start_turn=12,end_turn=14) reward=0.02 done=false error=null
[STEP] step=6 action=analyze_turn(turn_index=8,policy_hypothesis=unauthorized_commitment) reward=0.02 done=false error=null
[STEP] step=7 action=analyze_turn(turn_index=14,policy_hypothesis=failed_escalation) reward=0.02 done=false error=null
[STEP] step=8 action=flag_violation(violation_type=unauthorized_commitment,violation_severity=medium,turn_index=8) reward=0.00 done=false e
[STEP] step=9 action=flag_violation(violation_type=failed_escalation,violation_severity=high,turn_index=14) reward=0.00 done=false error=nu
[STEP] step=10 action=submit_report(compliance_pass=False) reward=0.92 done=true error=null
[END] success=true steps=10 score=0.923 rewards=0.05,0.05,0.02,0.02,0.02,0.02,0.02,0.00,0.00,0.92
[START] task=call_006 env=regtriage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=get_call_metadata() reward=0.05 done=false error=null
[STEP] step=2 action=get_sentiment_timeline() reward=0.05 done=false error=null
[STEP] step=3 action=read_transcript_chunk(start_turn=0,end_turn=2) reward=0.02 done=false error=null
[STEP] step=4 action=read_transcript_chunk(start_turn=5,end_turn=7) reward=0.02 done=false error=null
[STEP] step=5 action=analyze_turn(turn_index=7,policy_hypothesis=unauthorized_commitment) reward=0.02 done=false error=null
[STEP] step=6 action=flag_violation(violation_type=unauthorized_commitment,violation_severity=high,turn_index=7) reward=0.00 done=false err
[STEP] step=7 action=read_transcript_chunk(start_turn=13,end_turn=15) reward=0.02 done=false error=null
[STEP] step=8 action=read_transcript_chunk(start_turn=0,end_turn=15) reward=-0.02 done=false error=Chunk too large (16 turns). Maximum 5 tu
[STEP] step=9 action=submit_report(compliance_pass=False) reward=0.10 done=true error=null
[END] success=false steps=9 score=0.102 rewards=0.05,0.05,0.02,0.02,0.02,0.00,0.02,-0.02,0.10
[START] task=call_007 env=regtriage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=get_call_metadata() reward=0.05 done=false error=null
[STEP] step=2 action=get_sentiment_timeline() reward=0.05 done=false error=null
[STEP] step=3 action=read_transcript_chunk(start_turn=0,end_turn=2) reward=0.02 done=false error=null
[STEP] step=4 action=flag_violation(violation_type=regulatory_disclosure_failure,violation_severity=high,turn_index=None) reward=0.00 done=
[STEP] step=5 action=read_transcript_chunk(start_turn=2,end_turn=4) reward=0.02 done=false error=null
[STEP] step=6 action=flag_violation(violation_type=pii_exposure_risk,violation_severity=high,turn_index=4) reward=0.00 done=false error=nul
[STEP] step=7 action=analyze_turn(turn_index=4,policy_hypothesis=pii_exposure_risk) reward=0.02 done=false error=null
[STEP] step=8 action=submit_report(compliance_pass=False) reward=0.93 done=true error=null
[END] success=true steps=8 score=0.929 rewards=0.05,0.05,0.02,0.00,0.02,0.00,0.02,0.93
[START] task=call_008 env=regtriage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=get_call_metadata() reward=0.05 done=false error=null
[STEP] step=2 action=get_sentiment_timeline() reward=0.05 done=false error=null
[STEP] step=3 action=read_transcript_chunk(start_turn=0,end_turn=2) reward=0.02 done=false error=null
[STEP] step=4 action=read_transcript_chunk(start_turn=11,end_turn=13) reward=0.02 done=false error=null
[STEP] step=5 action=analyze_turn(turn_index=11,policy_hypothesis=unauthorized_commitment) reward=0.02 done=false error=null
[STEP] step=6 action=read_transcript_chunk(start_turn=14,end_turn=16) reward=0.02 done=false error=null
[STEP] step=7 action=read_transcript_chunk(start_turn=17,end_turn=18) reward=-0.02 done=false error=Invalid range [17, 18]. Valid: [0, 17].
[STEP] step=8 action=read_transcript_chunk(start_turn=17,end_turn=17) reward=0.02 done=false error=null
[STEP] step=9 action=analyze_turn(turn_index=11,policy_hypothesis=unauthorized_commitment) reward=0.02 done=false error=null
[STEP] step=10 action=read_transcript_chunk(start_turn=3,end_turn=7) reward=0.02 done=false error=null
[STEP] step=11 action=flag_violation(turn_index=6,violation_type=unauthorized_commitment,violation_severity=high) reward=0.00 done=false er
[STEP] step=12 action=read_transcript_chunk(start_turn=8,end_turn=12) reward=0.02 done=false error=null
[STEP] step=13 action=read_transcript_chunk(start_turn=13,end_turn=17) reward=0.65 done=true error=null
[END] success=true steps=13 score=0.746 rewards=0.05,0.05,0.02,0.02,0.02,0.02,-0.02,0.02,0.02,0.02,0.00,0.02,0.65
[START] task=call_009 env=regtriage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=get_call_metadata() reward=0.05 done=false error=null
[STEP] step=2 action=get_sentiment_timeline() reward=0.05 done=false error=null
[STEP] step=3 action=read_transcript_chunk(start_turn=0,end_turn=2) reward=0.02 done=false error=null
[STEP] step=4 action=flag_violation(violation_type=regulatory_disclosure_failure,violation_severity=high,turn_index=None) reward=0.00 done=
[STEP] step=5 action=read_transcript_chunk(start_turn=5,end_turn=7) reward=0.02 done=false error=null
[STEP] step=6 action=read_transcript_chunk(start_turn=11,end_turn=13) reward=0.02 done=false error=null
[STEP] step=7 action=flag_violation(violation_type=failed_escalation,violation_severity=high,turn_index=12) reward=0.00 done=false error=nu
[STEP] step=8 action=read_transcript_chunk(start_turn=21,end_turn=23) reward=-0.02 done=false error=Invalid range [21, 23]. Valid: [0, 22].
[STEP] step=9 action=read_transcript_chunk(start_turn=21,end_turn=22) reward=0.02 done=false error=null
[STEP] step=10 action=analyze_turn(turn_index=12,policy_hypothesis=unauthorized_commitment) reward=0.02 done=false error=null
[STEP] step=11 action=analyze_turn(turn_index=6,policy_hypothesis=unauthorized_commitment) reward=0.02 done=false error=null
[STEP] step=12 action=read_transcript_chunk(start_turn=14,end_turn=16) reward=0.02 done=false error=null
[STEP] step=13 action=analyze_turn(turn_index=16,policy_hypothesis=unauthorized_commitment) reward=0.02 done=false error=null
[STEP] step=14 action=read_transcript_chunk(start_turn=17,end_turn=19) reward=0.02 done=false error=null
[STEP] step=15 action=flag_violation(violation_type=unauthorized_commitment,violation_severity=high,turn_index=18) reward=0.00 done=false e
[STEP] step=16 action=submit_report(compliance_pass=False) reward=0.86 done=true error=null
[END] success=true steps=16 score=0.862 rewards=0.05,0.05,0.02,0.00,0.02,0.02,0.00,-0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.00,0.86
[START] task=call_010 env=regtriage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=get_call_metadata() reward=0.05 done=false error=null
[STEP] step=2 action=get_sentiment_timeline() reward=0.05 done=false error=null
[STEP] step=3 action=read_transcript_chunk(start_turn=0,end_turn=2) reward=0.02 done=false error=null
[STEP] step=4 action=read_transcript_chunk(start_turn=7,end_turn=9) reward=0.02 done=false error=null
[STEP] step=5 action=flag_violation(turn_index=8,violation_severity=high,violation_type=unauthorized_commitment) reward=0.00 done=false err
[STEP] step=6 action=read_transcript_chunk(start_turn=15,end_turn=17) reward=0.02 done=false error=null
[STEP] step=7 action=flag_violation(turn_index=16,violation_severity=medium,violation_type=incorrect_hold_procedure) reward=0.00 done=false
[STEP] step=8 action=read_transcript_chunk(start_turn=27,end_turn=29) reward=0.02 done=false error=null
[STEP] step=9 action=flag_violation(turn_index=16,violation_severity=high,violation_type=failed_escalation) reward=0.00 done=false error=null
[STEP] step=10 action=analyze_turn(turn_index=15,policy_hypothesis=pii_exposure_risk) reward=0.02 done=false error=null
[STEP] step=11 action=flag_violation(turn_index=15,violation_severity=high,violation_type=pii_exposure_risk) reward=0.00 done=false error=null
[STEP] step=12 action=submit_report(compliance_pass=False) reward=0.75 done=true error=null
[END] success=true steps=12 score=0.749 rewards=0.05,0.05,0.02,0.02,0.00,0.02,0.00,0.02,0.00,0.02,0.00,0.75
[START] task=call_011 env=regtriage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=get_call_metadata() reward=0.05 done=false error=null
[STEP] step=2 action=get_sentiment_timeline() reward=0.05 done=false error=null
[STEP] step=3 action=read_transcript_chunk(start_turn=0,end_turn=2) reward=0.02 done=false error=null
[STEP] step=4 action=flag_violation(violation_type=regulatory_disclosure_failure,violation_severity=high,turn_index=None) reward=0.00 done=false error=null
[STEP] step=5 action=read_transcript_chunk(start_turn=9,end_turn=11) reward=0.02 done=false error=null
[STEP] step=6 action=flag_violation(violation_type=unauthorized_commitment,violation_severity=high,turn_index=10) reward=0.00 done=false error=null
[STEP] step=7 action=analyze_turn(turn_index=9,policy_hypothesis=pii_exposure_risk) reward=0.02 done=false error=null
[STEP] step=8 action=flag_violation(violation_type=pii_exposure_risk,violation_severity=medium,turn_index=9) reward=0.00 done=false error=null
[STEP] step=9 action=submit_report(compliance_pass=False) reward=0.97 done=true error=null
[END] success=true steps=9 score=0.967 rewards=0.05,0.05,0.02,0.00,0.02,0.00,0.02,0.00,0.97
[START] task=call_012 env=regtriage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=get_call_metadata() reward=0.05 done=false error=null
[STEP] step=2 action=get_sentiment_timeline() reward=0.05 done=false error=null
[STEP] step=3 action=read_transcript_chunk(start_turn=0,end_turn=2) reward=0.02 done=false error=null
[STEP] step=4 action=flag_violation(violation_type=regulatory_disclosure_failure,violation_severity=high,turn_index=None) reward=0.00 done=false error=null
[STEP] step=5 action=read_transcript_chunk(start_turn=9,end_turn=11) reward=0.02 done=false error=null
[STEP] step=6 action=flag_violation(violation_type=unauthorized_commitment,violation_severity=high,turn_index=9) reward=0.00 done=false error=null
[STEP] step=7 action=read_transcript_chunk(start_turn=17,end_turn=19) reward=0.02 done=false error=null
[STEP] step=8 action=flag_violation(violation_type=failed_escalation,violation_severity=high,turn_index=18) reward=0.00 done=false error=null
[STEP] step=9 action=read_transcript_chunk(start_turn=20,end_turn=22) reward=-0.02 done=false error=Invalid range [20, 22]. Valid: [0, 21].
[STEP] step=10 action=read_transcript_chunk(start_turn=20,end_turn=21) reward=0.02 done=false error=null
[STEP] step=11 action=submit_report(compliance_pass=False) reward=0.69 done=true error=null
[END] success=true steps=11 score=0.690 rewards=0.05,0.05,0.02,0.00,0.02,0.00,0.02,0.00,-0.02,0.02,0.69

============================================================
EXECUTIVE DASHBOARD
============================================================
Model: Qwen/Qwen2.5-72B-Instruct
Environment: regtriage
------------------------------------------------------------
Easy   (4 tasks): avg=0.891 min=0.865 max=0.919
Medium (4 tasks): avg=0.675 min=0.102 max=0.929
Hard   (4 tasks): avg=0.817 min=0.690 max=0.967
------------------------------------------------------------
Overall: avg=0.794 | total_time=805.2s
============================================================
