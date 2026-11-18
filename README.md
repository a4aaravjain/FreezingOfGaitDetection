# Parkinson’s Freezing-of-Gait Early Warning Research

**Science Fair 2026–27 | Active development | Unfinished research prototype**

A few seconds of warning could make a meaningful difference to someone whose next step becomes unexpectedly difficult. This project investigates whether ankle motion sensors and machine learning can anticipate freezing of gait (FoG) in people with Parkinson’s disease, with the long-term goal of triggering a timely vibration cue.

The ambition is greater independence in everyday movement through a compact, accessible wearable. Reliable prediction, effective cueing, and real-world benefit remain research goals.

## Potential worldwide impact

The World Health Organization estimates that more than **8.5 million people lived with Parkinson’s disease in 2019**. That scale makes accessible mobility research consequential for patients, families, and care systems around the world. [Source: WHO](https://www.who.int/news-room/fact-sheets/detail/parkinson-disease)

This project specifically targets people who experience freezing of gait. The worldwide Parkinson’s population describes the broader need, rather than the number eligible for this prototype. If future studies establish reliable prediction and effective cueing, this approach could contribute to mobility support within a community numbering in the millions worldwide.

## Target audience

- **People with Parkinson’s who experience freezing of gait:** the intended future beneficiaries, with mobility and independence at the center of the project.
- **Families and caregivers:** whose daily lives are closely connected to the person’s ability and confidence to move.
- **Clinicians and rehabilitation researchers:** potential collaborators in evaluating warning timing, cueing, and meaningful patient outcomes.
- **Wearable technology and machine learning researchers:** exploring accessible sensing and generalization to unseen people.

## Research approach

The project explores ankle acceleration and gyroscope signals, short movement histories, and models that operate forward in time. The intended warning arrives seconds before freezing begins. The future wearable concept combines motion sensing, prediction, and vibration feedback.

The central challenge is identifying enough episodes while keeping false alarms low. The research includes participant separation, continuous-session replay, event-level scoring, verification tests, and reporting of unsuccessful experiments.

## Current status and evidence

**This project is not finished, has not met its research targets, and is not a validated medical device.**

The current continuous-session development summary reports **6 of 16 episodes predicted**, with **3 false alarms in approximately 11.54 monitored minutes**. Successful warnings averaged approximately **3.93 seconds before onset**. These results come from reused development recordings; hardware-delivered warning time has not been verified.

A subsequent published-architecture cascade study reported **15 of 72 episodes detected across 18 participants**, with **58 unsuccessful starts** and approximately **49.41 unsuccessful starts per known monitored hour**. It did not pass its promotion criteria. These studies have different evaluation contexts and should not be combined into one performance claim.

The evidence does not establish reduced falls, improved walking, effective vibration intervention, or reliable performance in new patients. No locked final assessment has been executed.

## Project scope

The project includes prediction pipelines, neural and non-neural experiments, sensor integration work, assessment code, tests, and experiment reports. Historical findings document the research process; older reports may use superseded scoring rules or session boundaries.

Current evidence appears in [CURRENT_RESEARCH_STATUS.json](CURRENT_RESEARCH_STATUS.json), [ASSESSMENT_PROTOCOL_STATUS.md](ASSESSMENT_PROTOCOL_STATUS.md), and [CONTINUOUS_SESSION_CORRECTION.md](CONTINUOUS_SESSION_CORRECTION.md). Attribution for adapted components appears in [THIRD_PARTY_NOTICES_PUBLISHED.md](THIRD_PARTY_NOTICES_PUBLISHED.md).

## Next milestones

Priorities are reducing false alarms, improving episode coverage across people, validating continuous operation, measuring the complete sensor-to-vibration delay, and conducting appropriately reviewed evaluation with genuinely unseen participants. Claims of patient benefit would require separate human evaluation.

**The vision is meaningful mobility support at global scale. The present contribution is an evolving, transparent research prototype working toward that goal.**

