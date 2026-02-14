# question_pool.py  — Clean, contiguous-by-instrument, validated

from collections import Counter

question_pool = []

def add_items(items):
    """Append items ensuring id uniqueness at build time."""
    seen = {q["id"] for q in question_pool}
    for it in items:
        if it["id"] in seen:
            raise ValueError(f"Duplicate id at build: {it['id']}")
        seen.add(it["id"])
        question_pool.append(it)

def mk_items(base, category, instrument):
    out = []
    for b in base:
        out.append({
            "id": b["id"],
            "category": category,
            "instrument": instrument,
            "text": b["text"],
            "options": b["options"],
            "score_range": b["score_range"],
        })
    return out


# =============================
# 1) DEPRESSION
# =============================

# PHQ-9 (partial subset from your list)
PHQ9_OPTIONS = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
PHQ9_SCORES  = [0, 1, 2, 3]
phq9 = mk_items([
    {"id": "phq9_1", "text": "Little interest or pleasure in doing things?", "options": PHQ9_OPTIONS, "score_range": PHQ9_SCORES},
    {"id": "phq9_2", "text": "Feeling down, depressed, or hopeless?", "options": PHQ9_OPTIONS, "score_range": PHQ9_SCORES},
    {"id": "phq9_9", "text": "Thoughts that you would be better off dead, or of hurting yourself?", "options": PHQ9_OPTIONS, "score_range": PHQ9_SCORES},
], "depression", "PHQ-9")
add_items(phq9)

# KADS-11 – Kutcher Adolescent Depression Scale (all 11 as provided)
KADS11_OPTS3 = ["Hardly ever", "Much of the time", "Most of the time"]
KADS11_SCO3  = [0, 1, 2]
kads11 = mk_items([
    {"id": "kads_1", "text": "Low mood, sadness, feeling blah or down, depressed, or just generally unhappy?", "options": KADS11_OPTS3, "score_range": KADS11_SCO3},
    {"id": "kads_2", "text": "Feeling tired, feeling fatigued, or having little energy?", "options": KADS11_OPTS3, "score_range": KADS11_SCO3},
    {"id": "kads_3", "text": "Feeling that you are a failure or not good enough or feeling very discouraged?", "options": KADS11_OPTS3, "score_range": KADS11_SCO3},
    {"id": "kads_4", "text": "Trouble concentrating, thinking, or making decisions?", "options": KADS11_OPTS3, "score_range": KADS11_SCO3},
    {"id": "kads_5", "text": "Feeling slowed down, having trouble getting going, or keeping up your usual pace?", "options": KADS11_OPTS3, "score_range": KADS11_SCO3},
    {"id": "kads_6", "text": "Feeling restless or fidgety, like you have to keep moving or can't sit still?", "options": KADS11_OPTS3, "score_range": KADS11_SCO3},
    {"id": "kads_7", "text": "Sleep problems can't sleep, waking up early, or sleeping too much?", "options": KADS11_OPTS3, "score_range": KADS11_SCO3},
    {"id": "kads_8", "text": "Feeling hopeless about the future?", "options": KADS11_OPTS3, "score_range": KADS11_SCO3},
    {"id": "kads_9", "text": "Feeling that life isn't worth living or thinking about death or suicide?", "options": KADS11_OPTS3, "score_range": KADS11_SCO3},
    {"id": "kads_10", "text": "Feeling irritable or angry?", "options": KADS11_OPTS3, "score_range": KADS11_SCO3},
    {"id": "kads_11", "text": "Avoiding friends, wanting to be alone more than usual?", "options": KADS11_OPTS3, "score_range": KADS11_SCO3},
], "depression", "KADS-11")
add_items(kads11)

# CES-D – Center for Epidemiologic Studies Depression Scale (20 as provided)
CESD_OPTS = ["Rarely or none of the time", "Some or a little of the time", "Occasionally or a moderate amount of time", "Most or all of the time"]
CESD_SCOR = [0, 1, 2, 3]
CESD_SCOR_REV = [3, 2, 1, 0]
cesd = mk_items([
    {"id": "cesd_1", "text": "I was bothered by things that usually don't bother me.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_2", "text": "I did not feel like eating; my appetite was poor.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_3", "text": "I felt that I could not shake off the blues even with help from my family or friends.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_4", "text": "I felt that I was just as good as other people.", "options": CESD_OPTS, "score_range": CESD_SCOR_REV},
    {"id": "cesd_5", "text": "I had trouble keeping my mind on what I was doing.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_6", "text": "I felt depressed.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_7", "text": "I felt that everything I did was an effort.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_8", "text": "I felt hopeful about the future.", "options": CESD_OPTS, "score_range": CESD_SCOR_REV},
    {"id": "cesd_9", "text": "I thought my life had been a failure.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_10", "text": "I felt fearful.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_11", "text": "My sleep was restless.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_12", "text": "I was happy.", "options": CESD_OPTS, "score_range": CESD_SCOR_REV},
    {"id": "cesd_13", "text": "I talked less than usual.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_14", "text": "I felt lonely.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_15", "text": "People were unfriendly.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_16", "text": "I enjoyed life.", "options": CESD_OPTS, "score_range": CESD_SCOR_REV},
    {"id": "cesd_17", "text": "I had crying spells.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_18", "text": "I felt sad.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_19", "text": "I felt that people dislike me.", "options": CESD_OPTS, "score_range": CESD_SCOR},
    {"id": "cesd_20", "text": "I could not get going.", "options": CESD_OPTS, "score_range": CESD_SCOR},
], "depression", "CES-D")
add_items(cesd)

# MFQ-SF – Mood and Feelings Questionnaire (Short Form)
MFQ_OPTS = ["Not true", "Sometimes", "True"]
MFQ_SCOR = [0, 1, 2]
mfqsf = mk_items([
    {"id": "mfqsf_1", "text": "I felt miserable or unhappy.", "options": MFQ_OPTS, "score_range": MFQ_SCOR},
    {"id": "mfqsf_2", "text": "I didn't enjoy anything at all.", "options": MFQ_OPTS, "score_range": MFQ_SCOR},
    {"id": "mfqsf_3", "text": "I felt so tired I just sat around and did nothing.", "options": MFQ_OPTS, "score_range": MFQ_SCOR},
    {"id": "mfqsf_4", "text": "I was very restless.", "options": MFQ_OPTS, "score_range": MFQ_SCOR},
    {"id": "mfqsf_5", "text": "I felt I was no good anymore.", "options": MFQ_OPTS, "score_range": MFQ_SCOR},
    {"id": "mfqsf_6", "text": "I cried a lot.", "options": MFQ_OPTS, "score_range": MFQ_SCOR},
    {"id": "mfqsf_7", "text": "I found it hard to think properly or concentrate.", "options": MFQ_OPTS, "score_range": MFQ_SCOR},
    {"id": "mfqsf_8", "text": "I hated myself.", "options": MFQ_OPTS, "score_range": MFQ_SCOR},
    {"id": "mfqsf_9", "text": "I was a bad person.", "options": MFQ_OPTS, "score_range": MFQ_SCOR},
    {"id": "mfqsf_10", "text": "I felt lonely.", "options": MFQ_OPTS, "score_range": MFQ_SCOR},
    {"id": "mfqsf_11", "text": "I thought nobody really loved me.", "options": MFQ_OPTS, "score_range": MFQ_SCOR},
    {"id": "mfqsf_12", "text": "I thought I could never be as good as other kids.", "options": MFQ_OPTS, "score_range": MFQ_SCOR},
    {"id": "mfqsf_13", "text": "I did everything wrong.", "options": MFQ_OPTS, "score_range": MFQ_SCOR},
], "depression", "MFQ-SF")
add_items(mfqsf)

# DASS-21 – Depression subscale (as provided)
DASS21_DEP_OPTS = ["Did not apply to me at all", "Applied to me to some degree", "Applied to me to a considerable degree", "Applied to me very much"]
DASS21_DEP_SCOR = [0, 1, 2, 3]
dass21_dep = mk_items([
    {"id": "dass_dep_1", "text": "I couldn't seem to experience any positive feeling at all.", "options": DASS21_DEP_OPTS, "score_range": DASS21_DEP_SCOR},
    {"id": "dass_dep_2", "text": "I found it difficult to work up the initiative to do things.", "options": DASS21_DEP_OPTS, "score_range": DASS21_DEP_SCOR},
    {"id": "dass_dep_3", "text": "I felt that I had nothing to look forward to.", "options": DASS21_DEP_OPTS, "score_range": DASS21_DEP_SCOR},
    {"id": "dass_dep_4", "text": "I felt down-hearted and blue.", "options": DASS21_DEP_OPTS, "score_range": DASS21_DEP_SCOR},
    {"id": "dass_dep_5", "text": "I was unable to become enthusiastic about anything.", "options": DASS21_DEP_OPTS, "score_range": DASS21_DEP_SCOR},
    {"id": "dass_dep_6", "text": "I felt I wasn't worth much as a person.", "options": DASS21_DEP_OPTS, "score_range": DASS21_DEP_SCOR},
    {"id": "dass_dep_7", "text": "I felt that life was meaningless.", "options": DASS21_DEP_OPTS, "score_range": DASS21_DEP_SCOR},
], "depression", "DASS-21")
add_items(dass21_dep)


# =============================
# 2) ANXIETY
# =============================

# DSM-5 Level 1 – Anxiety screen (3 as provided)
L1_OPTS5 = ["Not at all","Rare","Several days","More than half the days","Nearly every day"]
L1_SCOR5 = [0,1,2,3,4]
dsm_l1_anx = mk_items([
    {"id": "dsm_l1_11", "text": "Felt nervous, anxious, or scared?", "options": L1_OPTS5, "score_range": L1_SCOR5},
    {"id": "dsm_l1_12", "text": "Not been able to stop worrying?", "options": L1_OPTS5, "score_range": L1_SCOR5},
    {"id": "dsm_l1_13", "text": "Not been able to do things you wanted to, because they made you feel nervous?", "options": L1_OPTS5, "score_range": L1_SCOR5},
], "anxiety", "DSM-5 Level-1")
add_items(dsm_l1_anx)

# PROMIS Anxiety (Level 2) — items from your doc (4 provided here)
PROMIS_OPTS5 = ["Never","Almost Never","Sometimes","Often","Almost Always"]
PROMIS_SCOR5 = [1,2,3,4,5]  # per PROMIS anchors in your list
promis_l2 = mk_items([
    {"id": "anx_l2_01", "text": "Did you feel like something awful might happen?", "options": PROMIS_OPTS5, "score_range": PROMIS_SCOR5},
    {"id": "anx_l2_02", "text": "Did you feel nervous?", "options": PROMIS_OPTS5, "score_range": PROMIS_SCOR5},
    {"id": "anx_l2_03", "text": "Did you feel scared?", "options": PROMIS_OPTS5, "score_range": PROMIS_SCOR5},
    {"id": "anx_l2_13", "text": "Was it hard for you to relax?", "options": PROMIS_OPTS5, "score_range": PROMIS_SCOR5},
], "anxiety", "PROMIS-Anxiety-L2")
add_items(promis_l2)

# GAD-7 (keep one set; this is the canonical one)
GAD7_OPTS = ["Not at all","Several days","More than half the days","Nearly every day"]
GAD7_SCOR  = [0,1,2,3]
gad7 = mk_items([
    {"id": "gad7_1", "text": "Feeling nervous, anxious, or on edge?", "options": GAD7_OPTS, "score_range": GAD7_SCOR},
    {"id": "gad7_2", "text": "Not able to stop or control worrying?", "options": GAD7_OPTS, "score_range": GAD7_SCOR},
    {"id": "gad7_3", "text": "Worrying too much about different things?", "options": GAD7_OPTS, "score_range": GAD7_SCOR},
    {"id": "gad7_4", "text": "Trouble relaxing?", "options": GAD7_OPTS, "score_range": GAD7_SCOR},
    {"id": "gad7_5", "text": "Being so restless that it's hard to sit still?", "options": GAD7_OPTS, "score_range": GAD7_SCOR},
    {"id": "gad7_6", "text": "Becoming easily annoyed or irritable?", "options": GAD7_OPTS, "score_range": GAD7_SCOR},
    {"id": "gad7_7", "text": "Feeling afraid, as if something awful might happen?", "options": GAD7_OPTS, "score_range": GAD7_SCOR},
], "anxiety", "GAD-7")
add_items(gad7)

# SCARED (41 items) — from your enumerated list
SCARED_OPTS = ["Not True or Hardly Ever True", "Sometimes True", "Very True or Often True"]
SCARED_SCOR = [0, 1, 2]
SCARED_PAIRS = [
    ("SCARED_01","When you feel scared, does it feel hard for you to breathe?"),
    ("SCARED_02","Do you get headaches when you are at school?"),
    ("SCARED_03","Do you feel uncomfortable with people you don't know well?"),
    ("SCARED_04","Do you get scared if you have to sleep over at a friend's place or away from home?"),
    ("SCARED_05","Do you worry if other people like you?"),
    ("SCARED_06","When you get scared, do you feel like you might faint (हिन्दी: चक्कर आना; ಕನ್ನಡ: ತಲೆ ಸುತ್ತು)?"),
    ("SCARED_07","Do you feel nervous a lot?"),
    ("SCARED_08","Do you tend to stick close to your parents, following them around?"),
    ("SCARED_09","Do people tell you that you seem nervous?"),
    ("SCARED_10","Do you feel nervous around people you don't know well?"),
    ("SCARED_11","Do you get stomachaches at school?"),
    ("SCARED_12","When you get really scared, do you feel like you're going crazy?"),
    ("SCARED_13","Do you worry about sleeping by yourself?"),
    ("SCARED_14","Do you worry about whether you are as good as other kids?"),
    ("SCARED_15","When you get scared, do you feel like things around you aren't real?"),
    ("SCARED_16","Do you have bad dreams about something bad happening to your parents?"),
    ("SCARED_17","Do you worry about going to school?"),
    ("SCARED_18","When you feel scared, does your heart beat really fast?"),
    ("SCARED_19","Do you feel shaky or wobbly (हिन्दी: कांपना; ಕನ್ನಡ: ನಡುಗುವುದು)?"),
    ("SCARED_20","Do you have bad dreams about something bad happening to you?"),
    ("SCARED_21","Do you worry about things working out for you in the future?"),
    ("SCARED_22","When you get scared, do you sweat a lot (हिन्दी: पसीना; ಕನ್ನಡ: ಬೆವರು)?"),
    ("SCARED_23","Are you a person who worries a lot?"),
    ("SCARED_24","Do you get super scared sometimes for no reason at all?"),
    ("SCARED_25","Are you afraid to be alone in the house?"),
    ("SCARED_26","Is it hard for you to talk to people you don't know well?"),
    ("SCARED_27","When you get scared, do you feel like you're choking (हिन्दी: गला घुटना; ಕನ್ನಡ: ಕತ್ತು ಹಿಸುಕಿದ ಹಾಗೆ)?"),
    ("SCARED_28","Do people tell you that you worry too much?"),
    ("SCARED_29","Do you dislike being away from your family?"),
    ("SCARED_30","Are you afraid of having a panic attack (a sudden rush of intense fear)?"),
    ("SCARED_31","Do you worry that something bad might happen to your parents?"),
    ("SCARED_32","Do you feel shy with people you don't know well?"),
    ("SCARED_33","Do you worry about what's going to happen in the future?"),
    ("SCARED_34","When you get scared, do you feel like you have to throw up (हिन्दी: उल्टी; ಕನ್ನಡ: ವಾಂತಿ)?"),
    ("SCARED_35","Do you worry about how well you do things, like your schoolwork or hobbies?"),
    ("SCARED_36","Are you scared to go to school?"),
    ("SCARED_37","Do you worry about things that have already happened?"),
    ("SCARED_38","When you get scared, do you feel dizzy (हिन्दी: चक्कर; ಕನ್ನಡ: ತಲೆ ಸುತ್ತು)?"),
    ("SCARED_39","Do you feel nervous when you have to do something while people watch you (reading aloud, speaking in class, playing a sport)?"),
    ("SCARED_40","Do you feel nervous when you're going to parties or places where there will be people you don't know?"),
    ("SCARED_41","Do you feel shy?"),
]
add_items(mk_items(
    [{"id": pid, "text": txt, "options": SCARED_OPTS, "score_range": SCARED_SCOR} for pid, txt in SCARED_PAIRS],
    "anxiety", "SCARED"
))

# RCADS – Anxiety subset (as provided)
RCADS_OPTS = ["Never", "Sometimes", "Often", "Always"]
RCADS_SCOR = [0, 1, 2, 3]
RCADS_PAIRS = [
    ("RCADS_01","Do you worry about things (हिन्दी: चिंता; ಕನ್ನಡ: ಚಿಂತೆ)?"),
    ("RCADS_03","When you have a problem, do you get a funny feeling in your stomach?"),
    ("RCADS_04","Do you worry when you think you have done poorly at something?"),
    ("RCADS_05","Would you feel afraid of being on your own at home?"),
    ("RCADS_07","Do you feel scared when you have to take a test?"),
    ("RCADS_08","Do you feel worried when you think someone is angry with you?"),
    ("RCADS_09","Do you worry about being away from your parents?"),
    ("RCADS_10","Do you get bothered by bad or silly thoughts or pictures in your mind?"),
    ("RCADS_12","Do you worry that you will do badly at your school work?"),
    ("RCADS_13","Do you worry that something awful will happen to someone in your family?"),
    ("RCADS_14","Do you suddenly feel as if you can't breathe when there is no reason for this?"),
    ("RCADS_16","Do you have to keep checking that you have done things right (like a switch is off, or a door is locked)?"),
    ("RCADS_17","Do you feel scared if you have to sleep on your own?"),
    ("RCADS_18","Do you have trouble going to school in the mornings because you feel nervous or afraid?"),
    ("RCADS_20","Do you worry you might look foolish?"),
    ("RCADS_22","Do you worry that bad things will happen to you?"),
    ("RCADS_23","Does it seem like you can't get bad or silly thoughts out of your head?"),
    ("RCADS_24","When you have a problem, does your heart beat really fast?"),
    ("RCADS_26","Do you suddenly start to tremble or shake (हिन्दी: कांपना; ಕನ್ನಡ: ನಡುಗುವುದು) when there is no reason for this?"),
    ("RCADS_28","When you have a problem, do you feel shaky?"),
    ("RCADS_30","Do you worry about making mistakes?"),
    ("RCADS_31","Do you have to think of special thoughts (like numbers or words) to stop bad things from happening?"),
    ("RCADS_32","Do you worry what other people think of you?"),
    ("RCADS_33","Are you afraid of being in crowded places (like malls, movies, or buses)?"),
    ("RCADS_34","Do you suddenly feel really scared for no reason at all?"),
    ("RCADS_35","Do you worry about what is going to happen?"),
    ("RCADS_36","Do you suddenly become dizzy or faint (हिन्दी: चक्कर; ಕನ್ನಡ: ತಲೆ ಸುತ್ತು) when there is no reason for this?"),
    ("RCADS_38","Do you feel afraid if you have to talk in front of your class?"),
    ("RCADS_39","Does your heart suddenly start to beat too quickly for no reason?"),
    ("RCADS_41","Do you worry that you will suddenly get a scared feeling when there is nothing to be afraid of?"),
    ("RCADS_42","Do you have to do some things over and over again (like washing your hands, cleaning or putting things in a certain order)?"),
    ("RCADS_43","Do you feel afraid that you will make a fool of yourself in front of people?"),
    ("RCADS_44","Do you have to do some things in just the right way to stop bad things from happening?"),
    ("RCADS_45","Do you worry when you go to bed at night?"),
    ("RCADS_46","Would you feel scared if you had to stay away from home overnight?"),
    ("RCADS_47","Do you feel restless (हिन्दी: बेचैनी; ಕನ್ನಡ: ಅಸಹನತೆ)?"),
]
add_items(mk_items(
    [{"id": pid, "text": txt, "options": RCADS_OPTS, "score_range": RCADS_SCOR} for pid, txt in RCADS_PAIRS],
    "anxiety", "RCADS"
))

# SCAS (selected set from your list)
SCAS_OPTS = ["Never", "Sometimes", "Often", "Always"]
SCAS_SCOR = [0, 1, 2, 3]
SCAS_PAIRS = [
    ("SCAS_01","Do you worry about things?"),
    ("SCAS_02","Are you scared of the dark?"),
    ("SCAS_03","When you have a problem, do you get a funny feeling in your stomach?"),
    ("SCAS_04","Do you feel afraid?"),
    ("SCAS_05","Would you feel afraid of being on your own at home?"),
    ("SCAS_06","Do you feel scared when you have to take a test?"),
    ("SCAS_07","Do you feel afraid if you have to use public toilets or bathrooms?"),
    ("SCAS_08","Do you worry about being away from your parents?"),
    ("SCAS_09","Do you feel afraid that you will make a fool of yourself in front of people?"),
    ("SCAS_10","Do you worry that you will do badly at your school work?"),
    ("SCAS_12","Do you worry that something awful will happen to someone in your family?"),
    ("SCAS_13","Do you suddenly feel as if you can't breathe when there is no reason for this?"),
    ("SCAS_14","Do you have to keep checking that you have done things right (like the switch is off, or the door is locked)?"),
    ("SCAS_15","Do you feel scared if you have to sleep on your own?"),
    ("SCAS_16","Do you have trouble going to school in the mornings because you feel nervous or afraid?"),
    ("SCAS_18","Are you scared of dogs?"),
    ("SCAS_19","Does it seem like you can't get bad or silly thoughts out of your head?"),
    ("SCAS_20","When you have a problem, does your heart beat really fast?"),
    ("SCAS_21","Do you suddenly start to tremble or shake (हिन्दी: कांपना; ಕನ್ನಡ: ನಡುಗುವುದು) when there is no reason for this?"),
    ("SCAS_22","Do you worry that something bad will happen to you?"),
    ("SCAS_23","Are you scared of going to the doctors or dentists?"),
    ("SCAS_24","When you have a problem, do you feel shaky?"),
    ("SCAS_25","Are you scared of being in high places or lifts (elevators)?"),
    ("SCAS_27","Do you have to think of special thoughts (like numbers or words) to stop bad things happening?"),
    ("SCAS_28","Do you feel scared if you have to travel in the car, on a Bus or a train?"),
    ("SCAS_29","Do you worry what other people think of you?"),
    ("SCAS_30","Are you afraid of being in crowded places (shopping centres, movies, buses, busy playgrounds)?"),
    ("SCAS_32","Do you suddenly feel really scared for no reason at all?"),
    ("SCAS_33","Are you scared of insects or spiders?"),
    ("SCAS_34","Do you suddenly become dizzy or faint when there is no reason for this?"),
    ("SCAS_35","Do you feel afraid if you have to talk in front of your class?"),
    ("SCAS_36","Does your heart suddenly start to beat too quickly for no reason?"),
    ("SCAS_37","Do you worry that you will suddenly get a scared feeling when there is nothing to be afraid of?"),
    ("SCAS_39","Are you afraid of being in small closed places, like tunnels or small rooms?"),
    ("SCAS_40","Do you have to do some things over and over again (like washing your hands, cleaning or putting things in a certain order)?"),
    ("SCAS_41","Do you get bothered by bad or silly thoughts or pictures in your mind?"),
    ("SCAS_42","Do you have to do some things in just the right way to stop bad things happening?"),
    ("SCAS_44","Would you feel scared if you had to stay away from home overnight?"),
]
add_items(mk_items(
    [{"id": pid, "text": txt, "options": SCAS_OPTS, "score_range": SCAS_SCOR} for pid, txt in SCAS_PAIRS],
    "anxiety", "SCAS"
))

# Youth PSC-17 – Anxiety/Internalizing subset (as provided)
YPSC_OPTS = ["Never", "Sometimes", "Often"]
YPSC_SCOR = [0, 1, 2]
YPSC_PAIRS = [
    ("YPSC_01","Are you fidgety (हिन्दी: चुलबुला; ಕನ್ನಡ: ಚಂಚಲ), or unable to sit still?"),
    ("YPSC_02","Do you feel sad or unhappy?"),
    ("YPSC_03","Do you daydream too much?"),
    ("YPSC_05","Do you have trouble understanding other people's feelings?"),
    ("YPSC_06","Do you feel hopeless (हिन्दी: निराश; ಕನ್ನಡ: ನಿರಾಶೆ)?"),
    ("YPSC_07","Do you have trouble concentrating?"),
    ("YPSC_09","Do you feel down on yourself?"),
    ("YPSC_11","Do you seem to be having less fun?"),
    ("YPSC_13","Do you act as if you're driven by a motor, always on the go?"),
    ("YPSC_15","Do you worry a lot?"),
    ("YPSC_17","Do you get distracted easily?"),
]
add_items(mk_items(
    [{"id": pid, "text": txt, "options": YPSC_OPTS, "score_range": YPSC_SCOR} for pid, txt in YPSC_PAIRS],
    "anxiety", "Y-PSC-17"
))

# DASS-Y — Anxiety & Stress subset (as provided)
DASSY_OPTS = ["Not True", "A Little True", "Fairly True", "Very True"]
DASSY_SCOR = [0, 1, 2, 3]
DASSY_PAIRS = [
    ("DASSY_01","Did you get upset about little things?"),
    ("DASSY_02","Did you feel dizzy, like you were about to faint (हिन्दी: चक्कर आना; ಕನ್ನಡ: ತಲೆ ಸುತ್ತು)?"),
    ("DASSY_04","Did you have trouble breathing (e.g. fast breathing), even when you weren't exercising?"),
    ("DASSY_06","Did you find yourself over-reacting to situations?"),
    ("DASSY_07","Did your hands feel shaky (हिन्दी: कांपना; ಕನ್ನಡ: ನಡುಗುವುದು)?"),
    ("DASSY_08","Were you stressing about lots of things?"),
    ("DASSY_09","Did you feel terrified (हिन्दी: घबराहट; ಕನ್ನಡ: ಆತಂಕ)?"),
    ("DASSY_11","Were you easily irritated (हिन्दी: चिड़चिड़ा; ಕನ್ನಡ: ಕಿರಿಕಿರಿ)?"),
    ("DASSY_12","Did you find it difficult to relax?"),
    ("DASSY_14","Did you get annoyed when people interrupted you?"),
    ("DASSY_15","Did you feel like you were about to panic?"),
    ("DASSY_18","Were you easily annoyed?"),
    ("DASSY_19","Could you feel your heart beating really fast, even when you hadn't done any exercise?"),
    ("DASSY_20","Did you feel scared for no good reason?"),
]
add_items(mk_items(
    [{"id": pid, "text": txt, "options": DASSY_OPTS, "score_range": DASSY_SCOR} for pid, txt in DASSY_PAIRS],
    "anxiety", "DASS-Y"
))

# DASS-21 — Anxiety & Stress subset (as provided)
DASS21_AX_OPTS = ["Not at all", "To some degree", "Considerably", "Very much"]
DASS21_AX_SCOR = [0, 1, 2, 3]
DASS21_AX_PAIRS = [
    ("DASS21_01","Did you find it hard to relax or calm down?"),
    ("DASS21_02","Did you notice your mouth felt dry?"),
    ("DASS21_04","Did you have trouble breathing (like breathing too fast, or feeling out of breath without exercise)?"),
    ("DASS21_06","Did you tend to react too strongly to situations?"),
    ("DASS21_07","Did you feel shaky (for example, in your hands)?"),
    ("DASS21_08","Did you feel like you were using a lot of nervous energy?"),
    ("DASS21_09","Were you worried about situations where you might panic and embarrass yourself?"),
    ("DASS21_11","Did you find yourself getting agitated or restless (हिन्दी: बेचैन; ಕನ್ನಡ: ಚಡಪಡಿಕೆ)?"),
    ("DASS21_12","Did you find it difficult to relax?"),
    ("DASS21_14","Did you get annoyed easily by anything that stopped you from what you were doing?"),
    ("DASS21_15","Did you feel like you were close to panicking?"),
    ("DASS21_18","Did you feel like you were easily annoyed or sensitive?"),
    ("DASS21_19","Were you aware of your heart beating fast or skipping a beat, even without physical activity?"),
    ("DASS21_20","Did you feel scared without any good reason?"),
]
add_items(mk_items(
    [{"id": pid, "text": txt, "options": DASS21_AX_OPTS, "score_range": DASS21_AX_SCOR} for pid, txt in DASS21_AX_PAIRS],
    "anxiety", "DASS-21"
))

# LSAS-CA — two items per situation (Fear + Avoidance)
LSAS_FEAR_OPTS  = ["None", "Mild", "Moderate", "Severe"]
LSAS_FEAR_SCOR  = [0, 1, 2, 3]
LSAS_AVOID_OPTS = ["Never", "Occasionally", "Often", "Usually"]
LSAS_AVOID_SCOR = [0, 1, 2, 3]
LSAS_SITUATIONS = [
    ("LSAS_01","Talking to classmates on the phone."),
    ("LSAS_02","Participating in small groups in class."),
    ("LSAS_03","Eating in front of other people."),
    ("LSAS_04","Asking an adult you don't know well for help (like a shopkeeper)."),
    ("LSAS_05","Giving a presentation in class."),
    ("LSAS_06","Going to parties or school events."),
    ("LSAS_07","Writing on the board in front of the class."),
    ("LSAS_08","Talking with kids you don't know well."),
    ("LSAS_09","Starting a conversation with people you don't know well."),
    ("LSAS_10","Using school or public bathrooms."),
    ("LSAS_11","Walking into a room when others are already seated."),
    ("LSAS_12","Being the center of attention (like on your birthday)."),
    ("LSAS_13","Asking questions in class."),
    ("LSAS_14","Answering questions in class."),
    ("LSAS_15","Reading out loud in class."),
    ("LSAS_16","Taking tests."),
    ("LSAS_17","Saying \"no\" to people when they ask for something."),
    ("LSAS_18","Telling others you disagree with them."),
    ("LSAS_19","Looking people you don't know well in the eyes."),
    ("LSAS_20","Returning something to a store."),
    ("LSAS_21","Playing a sport or performing in front of others."),
    ("LSAS_22","Joining a new club or group."),
    ("LSAS_23","Meeting new people."),
    ("LSAS_24","Asking a teacher for permission to leave the classroom."),
]
lsas_items = []
for sid, stext in LSAS_SITUATIONS:
    lsas_items.append({"id": f"{sid}_fear", "text": f"Fear/Anxiety rating for: {stext}", "options": LSAS_FEAR_OPTS, "score_range": LSAS_FEAR_SCOR})
    lsas_items.append({"id": f"{sid}_avoid", "text": f"Avoidance rating for: {stext}", "options": LSAS_AVOID_OPTS, "score_range": LSAS_AVOID_SCOR})
add_items(mk_items(lsas_items, "anxiety", "LSAS-CA"))

# PDSS – Adapted (7 items)
PDSS_LIST = [
    ("PDSS_01","How many panic attacks or similar episodes did you have?",
     ["None", "Mild (no full attacks, few symptoms)", "Moderate (1-2 full attacks OR multiple smaller episodes)", "Severe (>2 full attacks)", "Extreme (attacks more than once a day)"], [0,1,2,3,4]),
    ("PDSS_02","How scary were they?",
     ["Not at all", "A little scary", "Moderately scary", "Severely scary", "Extremely scary"], [0,1,2,3,4]),
    ("PDSS_03","How much did you worry about having another one?",
     ["Not at all", "Occasionally", "Frequently", "Very often", "Nearly constantly"], [0,1,2,3,4]),
    ("PDSS_04","Did you avoid any places or situations because of this fear?",
     ["No avoidance", "Mild avoidance", "Moderate avoidance", "Severe avoidance", "Extreme avoidance"], [0,1,2,3,4]),
    ("PDSS_05","Did you avoid any activities (like exercise) because of this fear?",
     ["No avoidance", "Mild avoidance", "Moderate avoidance", "Severe avoidance", "Extreme avoidance"], [0,1,2,3,4]),
    ("PDSS_06","How much did this interfere with your schoolwork or home life?",
     ["No interference", "Slight interference", "Significant interference", "Substantial impairment", "Extreme impairment"], [0,1,2,3,4]),
    ("PDSS_07","How much did this interfere with your social life?",
     ["No interference", "Slight interference", "Significant interference", "Substantial impairment", "Extreme impairment"], [0,1,2,3,4]),
]
add_items(mk_items(
    [{"id": qid, "text": txt, "options": opts, "score_range": scor} for qid, txt, opts, scor in PDSS_LIST],
    "anxiety", "PDSS"
))


# =============================
# 3) OCD
# =============================

OCI_OPTS = ["Not at all", "A little", "Moderately", "A lot", "Extremely"]
OCI_SCOR = [0, 1, 2, 3, 4]
oci_r_pairs = [
    ("oci_r_1","I have saved up so many things that they get in the way."),
    ("oci_r_2","I check things more often than necessary."),
    ("oci_r_3","I get upset if objects are not arranged properly."),
    ("oci_r_4","I feel compelled to count while I am doing things."),
    ("oci_r_5","I find it difficult to control my own thoughts."),
    ("oci_r_6","I collect things I don't need."),
    ("oci_r_7","I repeatedly check doors, windows, drawers, etc."),
    ("oci_r_8","I get upset if others change the way I have arranged things."),
    ("oci_r_9","I feel I have to repeat certain numbers."),
    ("oci_r_10","I sometimes have to wash or clean myself simply because I feel contaminated."),
    ("oci_r_11","I am upset by unpleasant thoughts that come into my mind against my will."),
    ("oci_r_12","I avoid throwing things away because I am afraid I might need them later."),
    ("oci_r_13","I repeatedly check gas and water taps and light switches after turning them off."),
    ("oci_r_14","I need things to be arranged in a particular order."),
    ("oci_r_15","I feel that there are good and bad numbers."),
    ("oci_r_16","I wash my hands more often and longer than necessary."),
    ("oci_r_17","I frequently get nasty thoughts and have difficulty in getting rid of them."),
    ("oci_r_18","I am excessively concerned about germs and cleanliness."),
]
add_items(mk_items(
    [{"id": pid, "text": txt, "options": OCI_OPTS, "score_range": OCI_SCOR} for pid, txt in oci_r_pairs],
    "ocd", "OCI-R"
))

YBOCS_OPTS = ["None", "Mild", "Moderate", "Severe", "Extreme"]
YBOCS_SCOR = [0, 1, 2, 3, 4]
ybocs_pairs = [
    ("ybocs_1","Time spent on obsessions"),
    ("ybocs_2","Interference due to obsessions"),
    ("ybocs_3","Distress associated with obsessions"),
    ("ybocs_4","Resistance against obsessions"),
    ("ybocs_5","Degree of control over obsessions"),
    ("ybocs_6","Time spent on compulsions"),
    ("ybocs_7","Interference due to compulsions"),
    ("ybocs_8","Distress associated with compulsions"),
    ("ybocs_9","Resistance against compulsions"),
    ("ybocs_10","Degree of control over compulsions"),
]
add_items(mk_items(
    [{"id": pid, "text": txt, "options": YBOCS_OPTS, "score_range": YBOCS_SCOR} for pid, txt in ybocs_pairs],
    "ocd", "Y-BOCS"
))


# =============================
# 4) PTSD
# =============================

# PCL-5 (full, 20 as provided later)
PCL5_OPTS = ["Not at all", "A little bit", "Moderately", "Quite a bit", "Extremely"]
PCL5_SCOR = [0, 1, 2, 3, 4]
pcl5_pairs = [
    ("pcl5_1","Repeated, disturbing, and unwanted memories of the stressful experience?"),
    ("pcl5_2","Repeated, disturbing dreams of the stressful experience?"),
    ("pcl5_3","Suddenly feeling or acting as if the stressful experience were actually happening again (as if you were actually back there reliving it)?"),
    ("pcl5_4","Feeling very upset when something reminded you of the stressful experience?"),
    ("pcl5_5","Having strong physical reactions when something reminded you of the stressful experience (for example, heart pounding, trouble breathing, sweating)?"),
    ("pcl5_6","Avoiding memories, thoughts, or feelings related to the stressful experience?"),
    ("pcl5_7","Avoiding external reminders of the stressful experience (for example, people, places, conversations, activities, objects, or situations)?"),
    ("pcl5_8","Trouble remembering important parts of the stressful experience?"),
    ("pcl5_9","Having strong negative beliefs about yourself, other people, or the world?"),
    ("pcl5_10","Blaming yourself or someone else for the stressful experience or what happened after it?"),
    ("pcl5_11","Having strong negative feelings such as fear, horror, anger, guilt, or shame?"),
    ("pcl5_12","Loss of interest in activities that you used to enjoy?"),
    ("pcl5_13","Feeling distant or cut off from other people?"),
    ("pcl5_14","Trouble experiencing positive feelings (e.g., being unable to feel happiness or loving feelings)?"),
    ("pcl5_15","Irritable behavior, angry outbursts, or acting aggressively?"),
    ("pcl5_16","Taking too many risks or doing things that could cause you harm?"),
    ("pcl5_17","Being “superalert” or watchful or on guard?"),
    ("pcl5_18","Feeling jumpy or easily startled?"),
    ("pcl5_19","Having difficulty concentrating?"),
    ("pcl5_20","Trouble falling or staying asleep?"),
]
add_items(mk_items(
    [{"id": pid, "text": txt, "options": PCL5_OPTS, "score_range": PCL5_SCOR} for pid, txt in pcl5_pairs],
    "ptsd", "PCL-5"
))

# IES-R (22 items)
IESR_OPTS = ["Not at all", "A little bit", "Moderately", "Quite a bit", "Extremely"]
IESR_SCOR = [0, 1, 2, 3, 4]
iesr_pairs = [
    ("iesr_1","Any reminder brought back feelings about it."),
    ("iesr_2","I had trouble staying asleep."),
    ("iesr_3","Other things kept making me think about it."),
    ("iesr_4","I felt irritable and angry."),
    ("iesr_5","I avoided letting myself get upset when I thought about it or was reminded of it."),
    ("iesr_6","I thought about it when I didn't mean to."),
    ("iesr_7","I felt as if it hadn't happened or wasn't real."),
    ("iesr_8","I stayed away from reminders about it."),
    ("iesr_9","Pictures about it popped into my mind."),
    ("iesr_10","I was jumpy and easily startled."),
    ("iesr_11","I tried not to think about it."),
    ("iesr_12","I was aware that I still had a lot of feelings about it, but I didn't deal with them."),
    ("iesr_13","My feelings about it were kind of numb."),
    ("iesr_14","I found myself acting or feeling like I was back at that time."),
    ("iesr_15","I had trouble falling asleep."),
    ("iesr_16","I had waves of strong feelings about it."),
    ("iesr_17","I tried to remove it from my memory."),
    ("iesr_18","I had trouble concentrating."),
    ("iesr_19","Reminders of it caused me to have physical reactions, such as sweating, trouble breathing, nausea, or a pounding heart."),
    ("iesr_20","I had dreams about it."),
    ("iesr_21","I felt watchful or on-guard."),
    ("iesr_22","I tried not to talk about it."),
]
add_items(mk_items(
    [{"id": pid, "text": txt, "options": IESR_OPTS, "score_range": IESR_SCOR} for pid, txt in iesr_pairs],
    "ptsd", "IES-R"
))


# =============================
# 5) BIPOLAR
# =============================

# MDQ
MDQ_OPTS = ["Yes", "No"]
MDQ_SCOR = [1, 0]
mdq_pairs = [
    ("mdq_1","Has there ever been a period of time when you were not your usual self and... you felt so good or so hyper that other people thought you were not your normal self or were so hyper that you got into trouble?"),
    ("mdq_2","...you were so irritable that you shouted at people or started fights or arguments?"),
    ("mdq_3","...you felt much more self-confident than usual?"),
    ("mdq_4","...you got much less sleep than usual and found you didn't really miss it?"),
    ("mdq_5","...you were much more talkative or spoke much faster than usual?"),
    ("mdq_6","...thoughts raced through your head or you couldn't slow your mind down?"),
    ("mdq_7","...you were so easily distracted by things around you that you had trouble concentrating or staying on track?"),
    ("mdq_8","...you had much more energy than usual?"),
    ("mdq_9","...you were much more active or did many more things than usual?"),
    ("mdq_10","...you were much more social or outgoing than usual, for example, telephoning friends in the middle of the night?"),
    ("mdq_11","...you were much more interested in sex than usual?"),
    ("mdq_12","...you did things that were unusual for you or that other people might have thought were excessive, foolish, or risky?"),
    ("mdq_13","...spending money got you or your family into trouble?"),
]
add_items(mk_items(
    [{"id": pid, "text": txt, "options": MDQ_OPTS, "score_range": MDQ_SCOR} for pid, txt in mdq_pairs],
    "bipolar", "MDQ"
))
# MDQ co-occurrence and impairment items
add_items(mk_items([
    {"id": "mdq_14", "text": "If you checked YES to more than one of the above, have several of these ever happened during the same period of time?", "options": MDQ_OPTS, "score_range": MDQ_SCOR},
    {"id": "mdq_15", "text": "How much of a problem did any of these cause you – like being unable to work; having family, money, or legal troubles; getting into arguments or fights?", "options": ["No problem","Minor problem","Moderate problem","Serious problem"], "score_range": [0,1,2,3]},
], "bipolar", "MDQ"))

# BSDS
add_items(mk_items([
    {"id": "bsds_1", "text": "Please indicate how accurately the above paragraph describes you:", "options": ["This story fits me very well", "This story fits me fairly well", "This story fits me to some degree", "This story does not really describe me at all"], "score_range": [6,4,2,0]},
], "bipolar", "BSDS"))





# =============================
# Build-time sanity checks (optional)
# =============================

def _validate(pool):
    errs = []
    ids = [q["id"] for q in pool]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        errs.append(f"Duplicate IDs: {dupes}")

    for q in pool:
        if len(q["options"]) != len(q["score_range"]):
            errs.append(f"Len mismatch: {q['id']} options={len(q['options'])} scores={len(q['score_range'])}")
        if "instrument" not in q:
            errs.append(f"Missing instrument: {q['id']}")
        if "category" not in q:
            errs.append(f"Missing category: {q['id']}")
        if not isinstance(q["score_range"], list) or not all(isinstance(x, int) for x in q["score_range"]):
            errs.append(f"Scores must be int list: {q['id']}")

    if errs:
        raise AssertionError("Validation errors:\n - " + "\n - ".join(errs))

# Run validation on import (you can disable in production)
_validate(question_pool)


if __name__ == "__main__":
    import json
    with open("question_pool.json", "w", encoding="utf-8") as f:
        json.dump({"questions": question_pool}, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(question_pool)} questions to question_pool.json (with top-level 'questions' key).")
