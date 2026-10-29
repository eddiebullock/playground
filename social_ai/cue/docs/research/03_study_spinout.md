# Feasibility study design, regulation and spin-out route for an AI social-interaction tool for autistic adults

Prepared 2026-10-24. Scope: Cambridge PhD (Autism Research Centre, MRC iCASE with Emteq Labs); PREC ethics; browser/downloadable software, possibly a video-call companion and a post-hoc conversation/message interpreter.

Conventions: every claim carries a URL. "[Unverified]" marks items taken from secondary sources, from memory of the literature, or where the primary text could not be opened during this search. This is not legal advice; regulatory and IP points need confirmation with MHRA, the University Information Compliance Office, Research Operations Office (ROO) contracts team and Cambridge Enterprise.

---

## 1. Feasibility and acceptability methodology

### 1.1 Core frameworks (what to cite and how to use them)

| Framework | Use in this study | Source |
|---|---|---|
| Bowen et al. 2009, "How we design feasibility studies" | Eight focus areas: acceptability, demand, implementation, practicality, adaptation, integration, expansion, limited efficacy testing. Use as the organising frame for objectives. | https://pubmed.ncbi.nlm.nih.gov/19362699/ [citation from memory; verify page] |
| Sekhon, Cartwright & Francis 2017, Theoretical Framework of Acceptability (TFA) | Seven constructs: affective attitude, burden, ethicality, intervention coherence, opportunity costs, perceived effectiveness, self-efficacy. Use for both questionnaire and interview topic guide. | https://bmchealthservres.biomedcentral.com/articles/10.1186/s12913-017-2031-8 |
| Sekhon et al. 2022, generic TFA questionnaire | Eight items (one per construct plus one general acceptability item), prospective and retrospective versions. Think-aloud testing (n=20) found ethicality, intervention coherence and self-efficacy items were the ones people struggled with. Pilot the wording with autistic co-researchers before use, as these three constructs are the most abstract. | https://openaccess.city.ac.uk/id/eprint/27920/ ; https://doaj.org/article/97d57f869b2f45a6a742db2e33db6b0f |
| CONSORT 2010 extension to randomised pilot and feasibility trials (Eldridge et al. 2016) | Reporting checklist; if not randomised, still use it as a reporting guide (the authors say it is applicable to non-randomised feasibility studies in large part). | https://www.bmj.com/content/355/bmj.i5239 [URL from memory] |
| Eldridge et al. 2016 conceptual framework (PLoS ONE) | Feasibility is the umbrella; pilot studies are a subset. Lets you call a single-arm study a "feasibility study" legitimately. | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150205 [URL from memory] |
| MRC/NIHR complex interventions framework (Skivington et al. 2021) | Phases: develop, assess feasibility, evaluate, implement; core elements include context, programme theory, stakeholder engagement, key uncertainties, refinement, economic considerations. Frame the study as the "feasibility" phase and write a logic model/programme theory. | https://www.bmj.com/content/374/bmj.n2061 [URL from memory] |
| Progression criteria, traffic light (Avery et al. 2017) | Pre-specify red/amber/green thresholds for recruitment, adherence/engagement, retention and data completeness. | https://bmjopen.bmj.com/content/7/2/e013537 ; methods for powering against the red zone: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7856754/ |

Note: a CONSORT 2025 update of the main statement exists; whether the pilot extension has been updated was not checked [Unverified].

### 1.2 Sample-size rationale

- Julious 2005: 12 per group where there is no prior information, justified by feasibility and precision of mean/variance. https://eprints.whiterose.ac.uk/88744 (and see summary at https://pmc.ncbi.nlm.nih.gov/articles/PMC10662929)
- Teare et al. 2014 (Trials): at least 70 measured participants (35/group) to estimate an SD for a continuous outcome; around 60-100 per arm to estimate binary event rates. https://eprints.whiterose.ac.uk/88744/7/Sample%20size.pdf
- Hertzog 2008 (Res Nurs Health): 10-40 per group depending on aim; precision-based reasoning. https://pubmed.ncbi.nlm.nih.gov/18183564/ [citation from memory; not fetched]

Practical recommendation: for a single-arm feasibility study, justify N by precision of key proportions. For example, with N=30 completers a 70% engagement rate has a 95% CI of roughly +/-16 percentage points; with N=40 roughly +/-14 [standard binomial calculation, not sourced]. Inflate for expected 20-25% attrition (Molehill Mountain saw 76% complete the primary measure at about 15 weeks; see 1.5). A target of 36-45 enrolled is defensible for a single arm; if a two-arm (e.g. waitlist) feasibility RCT, 24-35 per arm aligns with Julious to Teare.

### 1.3 Suggested traffic-light progression criteria (illustrative, to adapt)

| Domain | Green (proceed) | Amber (amend) | Red (stop/rethink) |
|---|---|---|---|
| Recruitment | Target reached in planned window | 50-99% | <50% |
| Engagement (pre-defined minimum use, e.g. >=1 session/week for >=4 of 6 weeks) | >=70% | 50-69% | <50% |
| Retention to follow-up measures | >=75% | 60-74% | <60% |
| Acceptability (TFA general item, proportion rating acceptable) | >=70% | 50-69% | <50% |
| Safety/distress | No serious events attributable | Minor, manageable | Any serious attributable harm |

These thresholds follow the Avery et al. stop-amend-go structure (https://bmjopen.bmj.com/content/7/2/e013537); the specific numbers are a suggestion, not sourced.

### 1.4 Measures usable with autistic adults

| Construct | Measure | Evidence/notes | Source |
|---|---|---|---|
| Acceptability | TFA generic questionnaire (8 items) + TFA-structured interviews | See 1.1 | https://openaccess.city.ac.uk/id/eprint/27920/ |
| Usability | System Usability Scale (SUS, 10 items) | Used with autistic young people (SPAN-ASD, n=12); an adapted SUS has been used with neurodivergent users. Consider plain-language item rewording and avoid alternating positive/negative items confusing participants (or use the all-positive SUS variant). | https://doaj.org/article/bb4b544ca5734d6183fa994b518d0fd8 ; https://cirris.ulaval.ca/wp-content/uploads/SUS_UTAUT_adaptation_AAATE2025_8p_final.docx |
| User experience | UEQ / UEQ-S (26 / 8 items) | Used in a VR social-skills study with autistic participants; no app study with autistic adults found. | https://arxiv.org/pdf/2304.07498 |
| Effort/workload | NASA-TLX (6 subscales) | Not autism-validated [Unverified]; useful as a within-person comparison of "effort of an interaction with vs without tool". Pair with a single-item effort/exhaustion rating in EMA. | https://humansystems.arc.nasa.gov/groups/tlx/ [URL from memory] |
| Camouflaging | CAT-Q (25 items, alpha 0.94, test-retest r 0.77); CAT-Q short form 9 items (2024) | Developed with ARC co-authors; some construct-validity concerns for the Assimilation subscale. Relevant because a tool could increase or reduce masking effort - measure both directions. | https://discovery-pp.ucl.ac.uk/id/eprint/10060398 ; https://research-information.bris.ac.uk/en/publications/validation-of-the-camouflaging-autistic-traits-questionnaire-shor/ ; https://researchers.mq.edu.au/en/publications/exploring-the-construct-validity-of-the-camouflaging-autistic-tra/ |
| Quality of life | WHOQOL-BREF + WHO Disabilities module + ASQoL autism-specific items (McConachie et al. 2018, n=309) | Recommended together; note 8- vs 9-item discrepancy in reporting. | https://apps-eprint4-l.ncl.ac.uk/243200 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC8647037 |
| Loneliness | UCLA-3 / SELSA (Grace et al., n=203) or the modified UCLA-m (Love et al. 2025, n=628) | Autistic participants found standard scales hard to complete; UCLA-m is the autism-adapted option. | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11301961/ ; https://https-sage-cnpereading-com-443.webvpn1.xju.edu.cn/doi/10.1089/aut.2024.0258 [mirror URL; check DOI 10.1089/aut.2024.0258 at publisher] |
| Social anxiety | Few measures validated; DSM-5 Severity Measure for Social Anxiety Disorder did not hold a unidimensional structure in 284 autistic adults (Brett et al. 2024); LSAS-SR widely used but autism validation not found here | Treat as exploratory; consider using LSAS-SR with caveat. | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11301961/ (search result context) [Brett et al. citation unverified by fetch] |

Avoid using outcome measures as "efficacy" endpoints; in a feasibility study the question is whether they are completable, acceptable and variable enough (Bowen "limited efficacy testing").

### 1.5 EMA and diary designs

- Smartphone EMA is acceptable to autistic adults with average or above-average intelligence and language; systematic review of 32 studies (930 autistic participants) recommends adapting schedules, wording and incentives. https://discovery.ucl.ac.uk/id/eprint/10208515 [identified as the review via search; verify]
- Song et al. 2023: 40 autistic adults, once-daily survey for 30 days, mean response on about 27/30 days. https://ici.umn.edu/products/5Nhz9CiyQFuDDh-eybFKcg
- Design implication: an event-contingent "after a social interaction" diary (did you use the tool, how effortful was the interaction 0-10, how accurate/helpful was the interpretation, any distress) plus one daily end-of-day prompt is lower-burden than random signal-contingent sampling. Allow participants to choose prompt timing.

### 1.6 Adapting methods for autistic participants

- AASPIRE practice-based guidelines (Nicolaidis et al. 2019): transparency of goals, role clarity, power-sharing and fair pay for co-researchers; accessible consent, multiple modes of participation, adapted surveys and interview guides for participants. https://pmc.ncbi.nlm.nih.gov/articles/PMC6776684 ; https://journals.sagepub.com/doi/10.1177/1362361319830523
- Practical list (synthesis): written questions in advance; choice of interview mode (video, audio, text chat, email); camera optional; sensory-friendly in-person rooms; explicit timings and agendas; literal, unambiguous item wording; breaks; a named contact; opt-out without explanation.
- Compensation: NIHR public-involvement rate rose 10% to GBP 27.50/hour from 16 December 2025 (role-based range GBP 13.80-495). Use this for autistic co-researchers / advisory group. https://www.ucl.ac.uk/joint-research-office/news/2025/dec/nihr-increases-payment-rates-public-involvement-research ; https://www.sheffieldbrc.nihr.ac.uk/patient-public-involvement/staff-resources. Participant payments are separate (typically vouchers per session; check PREC norms). Warn participants that payments may interact with benefits [general NIHR guidance; Unverified for current wording].
- Critique to address up front: autistic-led and academic voices object to tools that train autistic people toward neurotypical norms; the double empathy problem (Milton) frames miscommunication as two-way. The Noora trial attracted exactly this criticism. Frame the tool as "translation/interpretation on the user's terms", measure ethicality (TFA) explicitly, and include an outcome on masking effort. https://www.taylorfrancis.com/chapters/oa-edit/10.4324/9781003189978-6/autism-double-empathy-problem-damian-milton-krysia-emily-waldock-nathan-keates ; https://autismanswersback.bearblog.dev/noora-the-ai-that-teaches-autistic-people-empathy-and-the-old-myth-it-recycles/ ; https://arxiv.org/html/2602.18962

### 1.7 Published feasibility studies to model on

| Study | Population / tool | Design and N | Key feasibility metrics | Source |
|---|---|---|---|---|
| Molehill Mountain (KCL/Autistica), Autism Research 2025/26 | Autistic people 16+ with anxiety; adapted CBT app | Single-arm, single site, 13 weeks; 100 enrolled of 123 screened in ~15 months | 76% completed primary measure at ~15 weeks; 65% adhered to full duration; mostly weekly use | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12853239/ ; protocol https://kclpure.kcl.ac.uk/portal/en/publications/molehill-mountain-feasibility-study-protocol-for-a-non-randomised/ |
| Brain in Hand (Tromans et al., BJPsych Open 2023) | Autistic adults (diagnosed or suspected), digital self-management app | 12-week prospective mixed-methods cohort, 7 NHS services | Improvements in anxiety and QoL; small, low-support-needs sample; company-linked | https://le.ac.uk/news/2023/may/brain-in-hand ; DOI 10.1192/bjo.2023.57 |
| Noora (Stanford), J Autism Dev Disord 2025/26 | Autistic 11-35 year olds; LLM chatbot for empathetic responses | RCT vs waitlist; 10 prompts/day, 5 days/week, 4 weeks (200 trials) | Reported improved empathetic responses; generalisation tested with conversation samples. N not confirmed here [Unverified]. Closest AI/LLM model. | https://pubmed.ncbi.nlm.nih.gov/39954218/ ; https://news.stanford.edu/stories/2025/08/ai-social-coach-noora-autism |
| SELFI (social media skills group) pilot RCT | Autistic adults; virtual group programme | Pilot RCT with waitlist, ~25-26 participants | Recruitment, attendance, fidelity supported feasibility; attrition and surveys supported acceptability | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11461777/ |
| MindChip (Tang et al. 2021) | Autistic adults; telehealth plus computer-based social-emotional training | Randomised pragmatic pilot, N=25 | Partial feasibility; need for engagement/personalisation | https://www.autismcrc.com.au/knowledge-centre/publications/development-and-feasibility-mindchip-social-emotional-telehealth |
| VR Job Interview Training (Smith et al. 2014) | Autistic adults; computer-based VR interview simulator | Single-blind RCT, N=26 (16 vs 10 TAU) | 90% session completion; rated easy and enjoyable | https://pmc.ncbi.nlm.nih.gov/articles/PMC4167908 |

Also relevant qualitative design evidence: autistic users' own ChatGPT practices (communication translation, emotional regulation) and risks (validation of paranoia, bias), e.g. https://dl.acm.org/doi/10.1145/3613904.3641989 and https://dl.acm.org/doi/full/10.1145/3706598.3713420. No published LLM feasibility trial specifically in autistic adults with standard feasibility metrics was found in this search; that gap is a selling point for the thesis.

Suggested design: mixed-methods, single-arm (or two-arm waitlist if resources allow), 6-8 weeks of real-world use, N=36-45 enrolled, TFA + SUS/UEQ-S + event-contingent diary + usage logs + pre/post CAT-Q, UCLA-m, ASQoL, and exit interviews analysed with TFA-framework analysis; autistic advisory group paid at NIHR rates; pre-registered progression criteria.

---

## 2. Regulation

### 2.1 When does the software become a medical device (UK)?

- Legal test: UK MDR 2002 reg. 2(1) defines a medical device by manufacturer intent, including software intended for "diagnosis, monitoring, treatment, alleviation of or compensation for an injury or handicap". https://www.legislation.gov.uk/uksi/2002/618/regulation/2
  - Important risk: an assistive tool aimed at autistic adults to overcome social-communication difficulties could be read as "compensation for a handicap", even without a treatment claim. This is the main qualification risk for this product and should be raised directly with MHRA (pre-submission advice) before commercial launch [interpretive; Unverified with MHRA].
- Intended purpose is set by labelling, instructions and promotional material; the MHRA intended-purpose guidance asks for structure/function, population, user and use environment. https://www.regdesk.co/blog/mhra-guidance-on-software-and-ai-as-a-medical-device/ ; https://www.burges-salmon.com/articles/102icot/mhra-publishes-guidance-for-software-and-ai-as-a-medical-device
- Digital mental health technology (DMHT) guidance (Feb 2025, v1.1; page shows later updates): a DMHT is SaMD if it has a medical purpose (performs a clinical task targeting clinical conditions/symptoms) and sufficient functionality; "supporting wellbeing" vs "aiding treatment of a mental health condition" is the dividing line; low-functionality products may not qualify even with a medical purpose. Contains worked examples (could not extract during this search; read pp. on examples). https://www.gov.uk/government/publications/digital-mental-health-technology-qualification-and-classification ; https://assets.publishing.service.gov.uk/media/6a68c3490825cc51a6c37c1d/Digital_mental_health_technology_device_characterisation__regulatory_qualification_and_classification.pdf ; https://inquisitiveminds.bristows.com/post/102jzqs/a-wellcome-development-for-samd-the-uk-publishes-guidance-on-digital-mental-hea
- Scope creep warning: adding AI mood analysis or clinician-facing outputs can push a wellbeing product into Class IIa/IIb. https://pharmaphorum.com/digital/digital-mental-health-technologies-and-changing-face-regulation
- MHRA has stated that even AI scribes are not automatically devices; medical purpose determines it. https://brodies.com/insights/healthcare-and-life-sciences/mhra-clarifies-that-ai-scribes-are-not-automatically-medical-devices/
- Research use: software used in a research study at Cambridge, not placed on the market, is generally not subject to UKCA marking, but if the study is a clinical investigation of a device (i.e. investigating medical performance/safety) MHRA notification and HRA routes apply [general principle; Unverified]. Keep the study framed as a feasibility/acceptability study of a communication-support tool, without diagnostic/treatment endpoints, and PREC remains the appropriate committee. PREC itself notes HRA review is normally required if participants are recruited via NHS services. https://www.bio.cam.ac.uk/files/prec_handbook.pdf

### 2.2 UKCA / CE status

- CE-marked devices are currently recognised in GB until 30 June 2030; MHRA consulted (Feb-Apr 2026) on indefinite recognition of EU MDR/IVDR devices and an international reliance route (US, Canada, Australia) potentially extending to SaMD; draft Medical Devices (Amendment) Regulations 2026 notified to WTO in May 2026. Outcome not confirmed. https://www.brownejacobson.com/about/news-media/mhra-consultation-indefinite-recognition-ce-marked-medical-devices ; https://www.s-ge.com/export/en/articles/guide/ukca-2026-market-access-uk-swiss-medtech-manufacturers
- Enhanced post-market surveillance in force since 16 June 2025 (serious incidents in 15 working days). https://www.regdesk.co/blog/mhra-guidance-on-software-and-ai-as-a-medical-device/
- Implication: if a later medical version is pursued, the EU MDR (Rule 11 typically makes decision-support software at least Class IIa) route plus GB recognition is likely the efficient path [Rule 11 point from memory; Unverified].

### 2.3 EU AI Act

- Article 5(1)(f): prohibits AI systems that infer emotions of natural persons in the workplace and education institutions, except for medical or safety reasons; in force since 2 February 2025. Commission Guidelines (4 Feb 2025) say medical uses such as autism diagnosis are recognised, but workplace "wellbeing" stress monitoring is not medical. Some commentary claims the medical exception needs a CE-marked device [Unverified in Act text]. https://fpf.org/blog/red-lines-under-eu-ai-act-unpacking-the-prohibition-of-emotion-recognition-in-the-workplace-and-education-institutions/ ; https://www.orrick.com/en/insights/2025/04/eu-commission-publishes-guidelines-on-the-prohibited-ai-practices-under-the-ai-act ; https://www.aigl.blog/ai-act-guidelines-on-prohibited-artificial-intelligence-ai-practices-an-analysis-for-the-emotion-recognition-field/
  - Direct consequence for the B2B/Access-to-Work channel: a video-call companion that infers colleagues' or the user's emotions from faces/voices at work would be prohibited in the EU unless it qualifies as medical. In the UK the AI Act does not apply domestically, but it applies if the product is placed on the EU market or its outputs are used in the EU.
  - "Emotion recognition" in the Act is defined by inference from biometric data. Inferring tone or intent from text (the post-hoc message interpreter) is generally outside that definition, and inference of intentions is reportedly not covered by the prohibition. https://fpf.org/blog/red-lines-under-eu-ai-act-unpacking-the-prohibition-of-emotion-recognition-in-the-workplace-and-education-institutions/ [text-exclusion point is from Guidelines commentary; Unverified against Guidelines]
- Annex III point 1(c): "AI systems intended to be used for emotion recognition" are high-risk (outside the prohibited contexts). https://artificialintelligenceact.eu/annex/3/
- Article 50(3): deployers of emotion recognition systems must inform exposed persons (relevant to bystanders on calls). https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50
- Timing: Digital Omnibus postponed Annex III high-risk obligations to 2 December 2027 and Annex I (product-embedded, e.g. medical devices) to 2 August 2028; Article 5 prohibitions not delayed. https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/ ; https://labs.cloudsecurityalliance.org/research/csa-research-note-eu-ai-act-omnibus-vii-deadline-delay-20260/
- Design implication: architect so that the core product is text/transcript-based interpretation chosen by the user (not facial/voice affect inference about others), with any face/voice affect features off by default, disabled in work contexts in the EU, and treated as a separate, possibly medical, module.

### 2.4 Data protection (UK GDPR / DPA 2018)

- Participants' autism status and any wellbeing data are health data, a special category (Art 9); research lawful basis at the University is typically public task (Art 6(1)(e)) plus Art 9(2)(j) research with safeguards [standard UK university position; Unverified for Cambridge wording]. Consent remains the ethical basis for participation.
- Biometric data is special category only when processed to uniquely identify a person; a plain voice recording is not biometric data until features are technically extracted; emotion classification sits outside the ICO's biometric recognition guidance, with a phase-two guidance on classification planned. https://ico.org.uk/about-the-ico/summary-of-responses-to-the-ico-consultation-on-draft-biometric-data-guidance-phase-one/ ; https://www.ropesgray.com/en/insights/viewpoints/102j5cq/ico-publishes-biometric-data-guidance ; https://www.gblock.app/articles/tilly-norwood-ai-hotline-face-scan-mood-2026
- Bystanders (other people on calls, people in pasted messages): they are data subjects. The household exemption may cover the individual user, but not the University (during research) or the company (as product), who are controllers or processors for anything sent to their servers. Transparency to non-users (Art 14) is hard, so minimise: process locally where possible, do not store raw audio/video, store only the participant's own reflections, redact names. Sharing or publishing recordings removes the household exemption. https://dataprotection.ie/sites/default/files/uploads/2024-09/Transcript-%20%E2%80%98Know%20Your%20Data%E2%80%99%20_CCTV%20in%20the%20home.pdf ; https://www.recordinglaw.com/united-kingdom/data-privacy/cctv-and-neighbours/
- DPIA: required where processing is likely high-risk; novel AI, special category data, vulnerable individuals and invisible processing of third parties all point to a DPIA for both the study and the product. Use the University's research DPIA process via the Information Compliance Office. ICO DPIA guidance: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/data-protection-impact-assessments-dpias/ [URL from memory]. Also review third-party LLM API terms (data retention, training use, transfers outside UK).

### 2.5 Recording calls (UK)

- Investigatory Powers Act 2016 s.3 offence is interception in the course of transmission without lawful authority; a party recording their own call is widely interpreted as not interception, but no authoritative confirmation found. https://www.legislation.gov.uk/ukpga/2016/25/section/3/enacted ; https://www.recordinglaw.com/united-kingdom/data-privacy/recording-phone-calls/
- Personal-use recording without telling the other party is generally lawful; sharing or publishing is where liability arises; businesses face stricter rules (2018 Business Interception Regulations require reasonable efforts to inform). https://www.sprintlaw.co.uk/articles/is-it-legal-to-record-conversations-without-consent-in-the-uk ; https://www.legislation.gov.uk/ukdsi/2018/9780111163603/note/data.htm
- For research: PREC is very unlikely to approve covert capture of non-consenting third parties. Recommended: (a) post-hoc interpreter using the participant's own typed/pasted summary or their own messages with third-party names redacted; (b) live companion only in calls where all parties have been told (a standard notice line) or in role-play calls with consenting researchers/actors; (c) no raw media retained. Also check Zoom/Teams terms on third-party bots [Unverified].

### 2.6 Wording intended purpose to stay outside device scope (research and early product)

Principles from the DMHT guidance and MDR definition: avoid naming a condition as the target, avoid clinical tasks (diagnose, treat, monitor, reduce symptoms), avoid "compensation for" a disability framing, keep functionality general-purpose, and keep marketing consistent with the label (MHRA reads promotional material as intended purpose). https://www.legislation.gov.uk/uksi/2002/618/regulation/2 ; https://www.gov.uk/government/publications/digital-mental-health-technology-qualification-and-classification

Draft statement (to test with MHRA/regulatory counsel; [Unverified that it is sufficient]):

> "[Product] is a general-purpose communication assistant that helps adults reflect on conversations and messages. It offers alternative possible readings of tone and intent, suggests ways to phrase replies, and helps users prepare for conversations. It is designed with and for neurodivergent people, but it is not intended to diagnose, treat, monitor, prevent or alleviate any medical condition or disability, and it does not replace professional advice."

Avoid: "reduces social anxiety", "improves social functioning in autism", "treats", "symptoms", "clinically proven", "detects emotions", outcome claims drawn from the thesis in marketing. Be aware that "designed for autistic people" plus "compensates for social-communication difficulty" is the grey zone; the safest commercial framing is universal design (useful to anyone, built neuro-inclusively). Keep study outcome measures (loneliness, social anxiety) clearly framed as research descriptors, not product claims.

---

## 3. Spin-out route at Cambridge

### 3.1 IP ownership (most decision-critical item)

- Cambridge default: students who are not employees own the IP they create, except where a studentship/sponsorship agreement or a University third-party agreement says otherwise. https://www.enterprise.cam.ac.uk/about/faq/ ; https://www.enterprise.cam.ac.uk/wp-content/uploads/2015/04/2005-IPR-Rights-Policy-Guidance-Mar09-Links-updated-August-2015.pdf
- iCASE: the partner agreement is expected to set out IPR and should be signed before the student starts; at other universities iCASE is explicitly a case where the University takes ownership. https://mrc-lid.lshtm.ac.uk/files/2022/09/CASE-information-for-prospective-supervisors.pdf ; https://registryservices.ed.ac.uk/sites/default/files/2026-08/Intellectual%20Property%20Policy%20Guidance%20for%20PGR%20Students.pdf
- Cambridge requires a collaboration agreement for all industry-partnered studentships (ROO contracts team). https://www.tech.cam.ac.uk/files/epsrc_industry_phd_funding_guide_feb_24_v2.pdf ; https://www.medschl.cam.ac.uk/mrc-doctoral-training-programme-icase-phd-studentships
- Typical iCASE terms (general pattern, not this agreement): partner gets a first option or non-exclusive/exclusive licence to foreground IP in its field, publication review/delay rights, and confidentiality over its background IP [Unverified; typical of Lambert-style collaboration agreements]. Emteq's background IP (sensor glasses, facial-activation sensing) likely stays with Emteq.
- Action: obtain and read the signed iCASE collaboration agreement now; map what is "foreground" (thesis outputs) vs new software written outside the project; get written clarity from Emteq and ROO before writing code intended for a company. Building the product on Emteq-independent architecture (browser/text first) reduces entanglement; alternatively, negotiate Emteq as partner/customer/shareholder.

### 3.2 Equity and process

- Cambridge Enterprise opt-in fast-track equity model (7 May 2024), fully dilutable pre-investment: 5% non-patented "Other"; 10% patented Other or non-patented Life Science (includes software for healthcare); 15% patented Life Science or non-patented Therapeutic; 20% patented Therapeutic. At 5% the University has no hold over founders' rights to non-patented Other technology and the company is still a Cambridge spin-out. Royalties to be structured not to drain early capital. https://www.enterprise.cam.ac.uk/news/new-fast-track-equity-model/
- Historical Cambridge average stake about 10% vs 22.8% nationally; national review recommends at most 10% for software. https://globalventuring.com/university/europe/uk-universities-spinout-equity-stakes/ ; https://www.enterprise.cam.ac.uk/for-the-university/independent-review-of-university-spin-out-companies-adoption-of-best-practices/
- Implication: a non-patented software tool for a wellbeing/neuro-inclusion market could arguably be "Other" (5%) rather than "Life Science" (10%); how it is categorised is a negotiating point. If the student owns the IP outright (no iCASE override), no licence may be needed at all, though the University may still have an interest in thesis materials and data.

### 3.3 Funding and support routes

| Route | What it offers | Fit / timing | Source |
|---|---|---|---|
| Innovate UK ICURe (Engage, Discover up to GBP 2.5k, Explore up to GBP 35k, Exploit) | Customer discovery, market validation; Discover prerequisite for Explore | Strong fit during PhD; Engage window closed 9 Oct 2026, watch next cycle | https://iuk-business-connect.org.uk/opportunities/icure-exploit/ ; https://iuk-business-connect.org.uk/wp-content/uploads/2025/09/EXJAN26-Key-Dates-Expectations-and-Guidance.pdf |
| Founders at the University of Cambridge (SYNC co-founder matching, SPARK, START pre-seed accelerator) | EIR mentorship, space | Good fit; check founders.cam.ac.uk for 2027 cohorts | https://founders.cam.ac.uk/100-experts-from-graphcore-arm-google-and-stripe-come-together-to-support-the-next-generation-of-cambridge-startups/ ; https://founders.cam.ac.uk/wp-content/uploads/2025/05/Founders-START2.0-Cohort-Digital-Lookbook.pdf |
| Accelerate Cambridge (Judge) | 11-12 week programme; teams of 2+ with Cambridge connection | Spring 2027 cohort 5 Apr-24 Jun 2027, deadline 15 Jan 2027 | https://www.jbs.cam.ac.uk/entrepreneurship/programmes/accelerate-cambridge/programme/apply/ |
| Cambridge Social Ventures / Social Venture Incubator (SVI) | Coaching, no equity; seed fund with Cambridge Enterprise up to GBP 20k for impact ventures | Good fit if impact-led | https://www.jbs.cam.ac.uk/faculty-research/centres/social-innovation/cambridge-social-ventures/social-venture-incubator/social-impact-seed-funding/ |
| Cambridge Enterprise Technology Investment Fund (proof of concept) and seed funds | TIF committed GBP 4M+ to 40 projects; UKRI PoC awards of GBP 250k | Later, once IP and team clear | https://www.enterprise.cam.ac.uk/news/proof-of-concept-funding-and-early-network-building-are-key-in-success-of-university-spinouts/ |
| RAEng Enterprise Fellowship | Up to GBP 75k equity-free, 12 months, training; PhD students eligible if viva before start date (check current call) | Ideal post-viva route | https://enterprisehub.raeng.org.uk/media/jmlniphb/enterprise-fellowships-guidance-july-2025.pdf |
| MRC Confidence in Concept / DPFS | CiC is an institutional award (apply via University); DPFS for larger academic-led development | CiC only if a medical (device) route is pursued | https://www.enterprise.cam.ac.uk/grant/mrc-biomedical-catalyst-developmental-pathway-funding-scheme-dpfs/ |
| Wellcome Leap | Large thematic programmes (2026: women's health) | Poor fit unless a matching programme opens | https://en.wikipedia.org/wiki/Wellcome_Leap ; https://wellcomeleap.org/?p=27130 |

### 3.4 Market size

- England: O'Nions et al. 2023 estimate 59-72% of autistic people undiagnosed (435,700-1,197,300); midpoint ~750,000 undiagnosed adults aged 20+ per UCL coverage, implying a total autistic population of roughly 1.2 million+ [derived]. https://www.ucl.ac.uk/news/2023/jun/number-autistic-people-england-may-be-twice-high-previously-thought ; https://research.gold.ac.uk/id/eprint/33269
- US: ~5.4 million adults (2.21%), CDC modelled estimate (Dietz et al. 2020); contested. https://disabilityscoop.com/?p=28327 ; https://www.ncsautism.org/blog/7g63q1d3zttpx9ye9u8r0hsed6fa8t
- Global: 61.8 million (1 in 127) in GBD 2021, all ages. https://pubmed.ncbi.nlm.nih.gov/39709974/
- Late-diagnosed adults are a growing, digitally active segment; self-identified and undiagnosed users widen the addressable market beyond diagnosis (which also argues for non-diagnosis-gated, non-medical positioning).

### 3.5 Comparables

| Company | Model | Funding/traction | Source |
|---|---|---|---|
| Tiimo (DK) | B2C subscription visual planner for neurodivergent users | USD 4.8M raised to Aug 2024; 50k+ paying subscribers | https://tech.eu/2024/08/28/neurodivergent-focused-app-tiimo-secures-16m-in-funding-round/ |
| Inflow (UK) | B2C ADHD CBT app subscription (~USD 200/yr) | USD 11M Series A, 14M+ total | https://tech.eu/2023/01/11/using-digital-cbt-to-help/ |
| Goblin Tools (BE) | Free web, cheap app, ~GBP 3/month Pro; includes "Formalizer" and "Judge" tone tools - closest functional competitor for message interpretation | Solo developer; free tier sets price expectations near zero | https://apps.apple.com/be/app/id6449003064 ; https://www.saner.ai/blogs/goblin-tools-review |
| Brain in Hand (UK) | B2B2C via NHS, local authorities, Access to Work | NHS SBRI funding; BJPsych Open evaluation | https://le.ac.uk/news/2023/may/brain-in-hand |
| Cognoa (US) | FDA-authorised autism diagnostic (paediatric) | ~USD 127M total reported [Unverified aggregator] | https://multiples.vc/private-comps/cognoa |
| Brain Power (US) | Google Glass social-skills apps for children | Small crowdfunding (USD 165k), DoD grant; limited scale - cautionary tale for hardware-dependent social AI | https://news.elearninginside.com/the-google-glass-based-autism-aid-by-brain-power-has-gone-to-market/ ; https://cdmrp.health.mil/arp/research_highlights/20ned_sahin_highlight |
| Auticon | Employment of autistic technologists (services) | 494 staff, 373 autistic, 287 clients (2025) | https://en.wikipedia.org/wiki/Auticon |
| Genius Within CIC / Lexxic | Neuroinclusion consultancy, workplace assessments, coaching; Lexxic on G-Cloud at GBP 1,000/day | Potential channel partners | https://geniuswithin.org/about-us/who-we-work-with/ ; https://www.applytosupply.digitalmarketplace.service.gov.uk/g-cloud-14/services/657780077936929 |
| Emteq Labs | Emotion-sensing eyewear (Sense), B2B partners only | Aggregator figures conflict (USD 0.6-3M) [Unverified] | https://insider.fitt.co/steen-strand-ceo-of-emteq-labs/ ; https://www.vcbacked.co/company/emteq |

Neuro-inclusive Slack/Teams plug-ins: no specific funded comparator found in this search [gap; worth a targeted search].

### 3.6 Business models and the Access to Work channel

- Access to Work: grants up to GBP 69,260 per person per year (2025-26), covering assistive software among other support; neurodivergent conditions eligible. https://www.gov.uk/government/publications/access-to-work-holistic-assessments-provider-guidance-for-contracts-effective-on-or-after-5-june-2023/chapter-1-introduction-and-overview-of-the-programme ; https://www.nao.org.uk/wp-content/uploads/2026/02/the-access-to-work-scheme-summary.pdf ; https://www.everway.com/en-gb/govt-grant-schemes/access-to-work
- But the channel is congested: average processing 109 days (Nov 2025) vs 25-day target; ~60-66k applications outstanding in 2026; mental health/learning-condition applicants rose to 51% of customers in 2024/25; scheme under government consultation and DWP hiring 480 staff to clear backlog by Sept 2027. https://www.disabilitynewsservice.com/new-access-to-work-applicants-being-told-they-could-have-to-wait-37-weeks-for-decision-mps-hear/ ; https://www.civilserviceworld.com/professions/article/dwp-to-hire-hundreds-of-new-staff-to-clear-disability-scheme-backlog ; https://rnid.org.uk/2026/02/access-to-work-why-reform-cant-wait/
- Getting onto assessors' recommendation lists (as Texthelp, Ayoa and CareScribe have done) requires a mature product, per-seat pricing and a reseller/training offering. https://website-us.texthelp.com/en-gb/resources/access-to-work-guide ; https://carescribe.io/blog/7-different-ways-the-access-to-work-grant-can-support-you-and-your-employees/
- Models: (1) B2C freemium subscription (low price ceiling set by Goblin Tools/ChatGPT); (2) B2B employer neuroinclusion licences via HR/DEI and consultancies (Genius Within, Lexxic); (3) Access to Work/DSA-funded seats; (4) later, NHS/clinical route only if pursuing device status.

### 3.7 Pitfalls

1. IP: unclear iCASE terms or Emteq rights can block investment; resolve first.
2. Commodity risk: general LLMs already do message tone interpretation for free; differentiation must be autistic co-design, privacy, trust and evidence.
3. Regulatory creep: autism-targeted marketing plus "compensation" framing may make it a device; EU workplace emotion inference is prohibited.
4. Workplace power: employer-purchased tools that analyse employees' interactions raise surveillance concerns (the rationale for Art 5(1)(f)); keep data user-owned and never reportable to employers.
5. Community acceptance: deficit/normalisation framing can trigger backlash (see Noora critique).
6. Access to Work delays lengthen B2B sales cycles; do not build the financial plan on it alone.
7. Hardware dependence (Brain Power lesson): software-first, device-agnostic.
8. Thesis vs company conflicts: publication timing vs confidentiality; separate research data from product data.

---

## Unverified / to confirm list

- Exact wording of Bowen, CONSORT pilot, Eldridge, Skivington, Hertzog URLs (from memory).
- MHRA DMHT worked examples text (PDF not machine-readable in this session).
- Whether autism-targeted assistive software is "compensation for a handicap" in MHRA's view.
- Commission Guidelines detail on text-based inference and the CE-marking condition for the medical exception.
- Contents of the actual Emteq iCASE collaboration agreement.
- Noora trial N; CONSORT 2025 pilot extension status; Cognoa and Emteq funding totals.
