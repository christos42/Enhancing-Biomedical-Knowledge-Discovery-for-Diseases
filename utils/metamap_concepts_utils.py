def get_entities(d):
    entities = {}
    for r in d.itertuples():
        if r.score < 0.4:
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
             'cui': "||".join(list(set((en1['cui'] + '||' + en2['cui']).split('||')))),
             'semantic_type': list(set(en1['semantic_type'] + en2['semantic_type'])),
             'position': str(chunk1[0] + 1) + '/' + str(chunk2[1] - chunk1[0]),
             'score': [en1['score'], en2['score']],
             'trigger': "||".join(list(set((en1['trigger'] + '||' + en2['trigger']).split('||')))),
             'mapped_semantic_type': list(set(en1['mapped_semantic_type'] + en2['mapped_semantic_type']))}

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
        try:
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
        except:
            if type(score1) == list:
                s1 = score1[0]
            else:
                s1 = score1
            if type(score2) == list:
                s2 = score2[0]
            else:
                s2 = score2
            if s1 > s2:
                keys_to_remove.append(p2)
            elif s1 < s2:
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


def resolve_overlaps_with_expansion(positions, d_):
    merged_entities = []
    keys_to_remove = []
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
                    ent1 = d_[p1]
                    ent2 = d_[p2]
                    chunk1 = get_chunk(p1)
                    chunk2 = get_chunk(p2)
                    # Merge the entities
                    f_m_ent = merge_sequent_entities(ent1, ent2, chunk1, chunk2)
                    merged_entities.append(f_m_ent)
                    keys_to_remove.append([p1, p2])

    return keys_to_remove, merged_entities



def check_expansion(position, sentence):
    p_start, p_stop = get_chunk(position)
    new_p_stop = p_stop
    for index in range(p_stop, len(sentence)):
        #if (sentence[index] in [' ', '(', ')', '<', '>']) or (sentence[index] == '.' and index == len(sentence) - 1):
        #if (sentence[index] in [' ', ',']) or (sentence[index] == '.' and index == len(sentence) - 1):
        if (sentence[index] in [' ']) or (sentence[index] == '.' and index == len(sentence) - 1):
            new_p_stop = index - 1
            break

    new_p_start = p_start
    index = p_start - 1
    while index >= 0:
        #if (sentence[index] in [' ', '(', ')', '<', '>']):
        if (sentence[index] in [' ']) or (index==0):
            new_p_start = index + 1
            break
        index -= 1

    if new_p_stop == p_stop and new_p_start == p_start:
        update = 0
    else:
        update = 1

    new_position = str(new_p_start + 1) + '/' + str(new_p_stop - new_p_start + 1)
    return update, new_position


def expand_entities(entities, sentence):
    updated_dict = {}
    for k in entities:
        update, new_position = check_expansion(k, sentence)
        if update == 1:
            updated_dict[new_position] = entities[k].copy()
            updated_dict[new_position]['position'] = new_position
        else:
            updated_dict[k] = entities[k].copy()
    return updated_dict