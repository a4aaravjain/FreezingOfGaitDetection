"""Compare matched development replays before spending time on streaming verification."""
def compare_candidates(previous,previous_events,current,current_events,tolerance=1e-6):
 reasons=[];improvements=[]
 for key in ['total_false_alarms','censored_alarm_starts','false_alarm_seconds','maximum_alarm_duration_seconds']:
  if current[key]>previous[key]+tolerance:reasons.append(key+' increased')
 if current['total_fog_episodes']!=previous['total_fog_episodes']:reasons.append('episode denominator changed')
 if abs(current['monitored_seconds']-previous['monitored_seconds'])>tolerance:reasons.append('scored exposure changed')
 old={(e['segment'],e['onset']):e['warning_seconds'] for e in previous_events if e['warning_seconds'] is not None}
 new={(e['segment'],e['onset']):e['warning_seconds'] for e in current_events if e['warning_seconds'] is not None}
 if not set(old).issubset(new):reasons.append('previously warned episode lost')
 for key,lead in old.items():
  if key in new and new[key]<lead-tolerance:reasons.append('previous warning moved later')
  if key in new and new[key]>lead+tolerance:improvements.append('earlier warning')
 if len(new)>len(old):improvements.append('additional warned episode')
 if any(lead<2-tolerance for lead in new.values()):reasons.append('less than two seconds lead')
 if not improvements:reasons.append('no episode or lead improvement')
 return dict(eligible_for_streaming_verification=not reasons,reasons=sorted(set(reasons)),improvements=sorted(set(improvements)),independent_validation=False)
