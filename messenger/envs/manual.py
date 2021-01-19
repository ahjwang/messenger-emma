import random
import json

class TextManual:
    '''
    Temporary drop-in class with get_document() same as MessengerBaseTemplates
    except use json of free-form text instead of templates.
    TODO: Merge the free-form and template text classes.
    '''
    def __init__(self, json_path):
        # TODO: code for json file verification
        with open(json_path, "r") as f:
            self.descriptors = json.load(f)

    def get_descriptor(self, entity=None, role=None, entity_type=None, no_type_p=0.15):
        '''
        Get a descriptor using the templates.
        Parameters:
        entity: The object that is being described (e.g. alien)
        role: The role the the object plays in the env. (e.g. enemy)
        entity_type:
            The object type (e.g. chaser)
        no_type_p:
            The probability of returning a descriptor that does not have
            any type information (only if entity_type is not None).
        '''
        if random.random() < no_type_p: # no type information
            return random.choice(self.descriptors[entity][role]["unknown"])

        else:
            return random.choice(self.descriptors[entity][role][entity_type])

    def get_document(self, enemy, message, goal, shuffle=True, 
                enemy_type=None, message_type=None, goal_type=None,
                append=False, delete=False, **kwargs):
        '''
        Makes a document for Messenger using the specified parameters.
        If no type is provided, a random type will be selected.

        Parameters:
        append: 
            If True, append an extraneous sentence to the document describing a 
            random object that is not in {enemy, message, goal}.
        delete: If True, Delete a random descriptor from the document.
        shuffle: 
            If True, shuffles the order of the descriptors
        kwargs:
            All other kwargs go to get_descriptor()
        '''

        document = [
            self.get_descriptor(entity=enemy, entity_type=enemy_type, role="enemy", **kwargs),
            self.get_descriptor(entity=message, entity_type=message_type, role="message", **kwargs),
            self.get_descriptor(entity=goal, entity_type=goal_type, role="goal", **kwargs)
        ]

        if delete: # delete a random descriptor
            document = random.sample(document, 2)

        if append:
            # choose an object not in {enemy, message, goal}
            valid_objs = [obj.name for obj in defaults.NPCS if obj.name not in [enemy, message, goal]]
            rand_obj = random.choice(valid_objs)
            result = None
            while result is None:
                try:
                    result = self.get_descriptor(entity=rand_obj, **kwargs)
                except:
                    pass
            document.append(result)
            
        if shuffle:
            document = random.sample(document, len(document))

        return document