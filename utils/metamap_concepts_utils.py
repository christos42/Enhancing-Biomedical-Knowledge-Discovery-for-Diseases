def get_entities(d):
    entities = {}
    for r in d.itertuples():
        if r.score < 0.9:
            continue
        pos = r.pos_info.split(';')
        for p in pos:
            if p not in entities:
                entities[p] = {'preferred_name': r.preferred_name,
                               'cui': r.cui,
                               'semantic_type': r.semtypes,
                               'position': p,
                               'score': r.score,
                               'trigger': r.trigger}

    return entities


def get_chunk(pos):
    start = int(pos.split('/')[0]) - 1
    stop = start + int(pos.split('/')[1])
    return [start, stop]


def merge_sequent_entities(en1, en2, chunk1, chunk2):
    m_ent = {'preferred_name': en1['preferred_name'] + '||' + en2['preferred_name'],
             'cui': en1['cui'] + '||' + en2['cui'],
             'semantic_type': en1['semantic_type'] + en2['semantic_type'],
             'position': str(chunk1[0]) + '/' + str(chunk2[1] - chunk1[0] + 1),
             'score': [en1['score'], en2['score']],
             'trigger': en1['trigger'] + '||' + en2['trigger'],
             'mapped_semantic_type': en1['mapped_semantic_type'] + en2['mapped_semantic_type']}
    return m_ent


def detect_overlaps(positions, d_):
    overlaps = []
    for i1, p1 in enumerate(positions):
        for i2, p2 in enumerate(positions):
            if i1 == i2:
                continue
            else:
                p1_start = int(p1.split('/')[0]) - 1
                p1_stop = p1_start + int(p1.split('/')[1])
                p2_start = int(p2.split('/')[0]) - 1
                p2_stop = p2_start + int(p2.split('/')[1])
                if (p1_start <= p2_start) and (p2_start <= p1_stop):
                    cui1 = d_[p1]['cui']
                    cui2 = d_[p2]['cui']
                    if cui1 == cui2:
                        flag = 0
                        for i3, o in enumerate(overlaps):
                            if i1 in o:
                                flag = 1
                                overlaps[i3].append(i2)
                            elif i2 in o:
                                flag = 1
                                overlaps[i3].insert(o.index(i2), i1)
                        if flag == 0:
                            overlaps.append([i1, i2])
        
    return overlaps


def resolve_overlaps(positions, d_, overlaps):
    keys_to_remove = []
    for o in overlaps:
        p1 = positions[o[0]]
        p2 = positions[o[1]]
        score1 = d_[p1]['score']
        score2 = d_[p2]['score']
        if score1 > score2:
            keys_to_remove.append(p2)
        elif score1 < score2:
            keys_to_remove.append(p1)
        else:
            p1_start = int(p1.split('/')[0]) - 1
            p1_stop = p1_start + int(p1.split('/')[1])
            p2_start = int(p2.split('/')[0]) - 1
            p2_stop = p2_start + int(p2.split('/')[1])
            len1 = p1_stop - p1_start
            len2 = p2_stop - p2_start
            if len1 > len2:
                keys_to_remove.append(p2)
            else:
                keys_to_remove.append(p1)

    return keys_to_remove


def check_expansion(position, sentence):
    p_start, p_stop = get_chunk(position)
    new_p_stop = p_stop
    for index in range(p_stop, len(sentence)):
        if (sentence[index] in [' ', '(', ')', '<', '>']) or (sentence[index] == '.' and index == len(sentence)):
            new_p_stop = index - 1
            break

    if new_p_stop != p_stop:
        update = 1
    else:
        update = 0

    new_position = str(p_start + 1) + '/' + str(new_p_stop - p_start + 1)
    return update, new_position


def expand_entities(entities, sentence):
    updated_dict = {}
    for k in entities:
        update, new_position = check_expansion(k, sentence)
        if update == 1:
            updated_dict[new_position] = entities[k]
            updated_dict[new_position]['position'] = new_position
        else:
            updated_dict[k] = entities[k]
    return updated_dict