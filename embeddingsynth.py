import torch

def func2method(self, func):
    if hasattr(self.__class__, func.__name__) and getattr(self.__class__, func.__name__) is func:
        return getattr(self, func.__name__)
    else:
        return func

class EmbeddingGeneration:
    def __init__(self, generation_model, tokenizer):
        if type(generation_model) is str:
            import transformers.models.auto as auto
            #config = auto.tokenization_auto.AutoConfig.from_pretrained(generation_model)
            generation_model = auto.modeling_auto.AutoModelForCausalLM.from_pretrained(generation_model)
        if type(tokenizer) is str:
            import transformers.models.auto as auto
            tokenizer = auto.tokenization_auto.AutoTokenizer.from_pretrained(tokenizer)
        self.model = generation_model
        if torch.has_cuda:
            self.model = self.model.cuda()
        self.tokenizer = tokenizer
        if hasattr(self.model, 'wte'):
            self.embed = self.model.wte
        else:
            self.embed = self.model.transformer.wte

        for token_name in 'BOS CLS EOS MASK PAD SEP UNK'.split(' '):
            token_id = getattr(self.tokenizer, token_name.lower() + '_token_id')
            if token_id is not None:
                token_id_tensor = torch.tensor(token_id, device = self.model.device)
                token_embeds = self.ids2embeds(token_id_tensor)
            else:
                token_embeds = None
            setattr(self, token_name, token_embeds)
        
        #if hasattr(self.model, 'wpe'):
        #    self.pembed = self.model.wpe
        #    wpe_spot = (self.model, 'wpe')
        #    #self.model = WrappedObject(self.model, wpe = wrap_wpe)
        #else:
        #    self.pembed = self.model.transformer.wpe
        #    wpe_spot = (self.model.transformer, 'wpe')
        #    #self.model = WrappedObject(self.model, transformer = WrappedObject(self.model.transformer, wpe = wrap_wpe))
        #class WrappedEmbedding(torch.nn.Module):
        #    def __init__(self, wrappee):
        #        super().__init__()
        #        self.wrappee = wrappee
        #        self.inputs_by_positions = {}
        #    def prepare(self, inputs_embeds, positions_embeds):
        #        self.inputs_by_positions[positions_embeds.data_ptr()] = inputs_embeds
        #    def forward(self, position_ids):
        #        input_embeds = self.inputs_by_positions.pop(position_ids.data_ptr(), None)
        #        if position_ids.dtype.is_floating_point:
        #            return position_ids.view(-1, input_embeds.shape[-2], self.wrappee.embedding_dim)
        #        else:
        #            return self.wrappee(position_ids)
        #self.pembed_wrapper = WrappedEmbedding(self.pembed)
        #setattr(wpe_spot[0], wpe_spot[1], self.pembed_wrapper)

    @property
    def dim(self):
        return self.embed.embedding_dim

    @property
    def tokens(self):
        return self.embed.num_embeddings

    def greedy(self, logits):
        return self.ids2embeds(self.logits2ids(logits))
    def sum(self, logits):
        return (self.logits2probs(logits) @ self.embed.weight)

    def logits2ids(self, logits):
        return logits.argmax(dim=-1)
    def logits2probs(self, logits):
        return logits.softmax(dim=-1)
    def logits2strs(self, logits):
        return self.ids2strs(self.logits2ids(logits))
    def ids2embeds(self, ids):#, position=False):
        #if position:
        #    return self.pembed(ids)
        #else:
            return self.embed(ids)
    def ids2strs(self, ids):
        if len(ids.shape) > 1:
            return self.tokenizer.batch_decode(ids)
        else:
            return self.tokenizer.decode(ids)
    def strs2ids(self, strs):
        if type(strs) is str:
            ids = self.tokenizer.encode(strs, return_tensors='pt')[0]
        else:
            ids = self.tokenizer.batch_encode_plus(strs, return_tensors='pt')['input_ids']
        if torch.has_cuda:
            ids.cuda()
        return ids
    def logits2topk(self, logits=None, k=5, probs=False, str=True):
        if logits is None:
            return lambda logits: self.logits2topids(logits, k, probs)
        if k is None:
            elems = logits.sort(dim=-1)
        else:
            elems = logits.topk(dim=-1,k=k)
        if str:
            values = self.ids2strs(elems.indices.view(-1, 1))
            # convert values to logits.shape[:-1],k
            def list_view(list, shape, start, size):
                if len(shape) == 1:
                    return list[start:start+size]
                # for innermost, a subrange is returned
                # for after that, a number of subranges are connected
                dim = shape[0]
                offset = size // dim
                return [
                    list_view(list[suboffset:suboffset + offset], shape[1:], suboffset, offset)
                    for suboffset in range(start, start + size, offset)
                ]
            values = list_view(values, (*logits.shape[:-1],k), 0, len(values))
        else:
            values = elems.indices
        if probs:
            probs = self.logits2probs(elems.values)
            return [*zip(values, probs)]
        else:
            return values
    def logits2logits(self, logits):
        return logits

    def cat(self, *parts, shape = None, dim =-2):
        if shape is None:
            shape = [1] * max((len(part.shape) for part in parts))
            for part in parts:
                for idx in range(-len(part.shape), 0):
                    if idx != dim:
                        shape[idx] = max(shape[idx], part.shape[idx])
        else:
            shape = [*shape]
        reshaped = []
        for idx, part in enumerate(parts):
            while len(part.shape) < len(shape):
                part = part[None,...]
            shape[dim] = part.shape[dim]
            part = part.expand(shape)
            reshaped.append(part)
        return torch.cat(reshaped, dim=dim)

    def prefix(self, prefix, embeds, dim = -2):
        return self.cat(prefix, embeds, shape = embeds.shape, dim = dim)

    def suffix(self, embeds, suffix, dim = -2):
        return self.cat(embeds, suffix, shape = embeds.shape, dim = dim)

    def process(self, embeds, output=logits2strs, prefix = True):
        embeds = self.inputs2embeds(embeds)
        output = func2method(self, output)

        if prefix:
            if prefix is True:
                prefix = self.BOS
            embeds = self.prefix(prefix, embeds)

        output = func2method(self, output)
        model_logits = self.model(inputs_embeds = embeds).logits
       
        return output(model_logits)

    # each pass through a model produces a distribution of tokens for each known token, and the next unknown token
    def generate(self, embeds, size=None, logits2embeds=greedy, output=logits2strs, handler=None):#stop_sequence=None, handler=None): #pembeds=None
        
        # passing in the position embeddings, which are just more embeddings summed with the input embeddings, may for gpt2 require processing the transformer layers manually.

        #ids = self.inputs2ids(embeds)
        #if stop_sequence is not None:
        #    stop_sequence = self.inputs2ids(stop_sequence)
        #embeds, pembeds = self.inputs2embeds(embeds, pembeds)
        embeds = self.inputs2embeds(embeds)
        #logits = torch.empty(*embeds.shape[:-2],0,self.embed.num_embeddings,device=self.model.device)
        logits2embeds = func2method(self, logits2embeds)
        output = func2method(self, output)
        while True:
            if size is not None and embeds.shape[-2] >= size:
                break
            #if stop_sequences is not None:
            #     and ids[:-stop_sequence.shape[-1]] == stop_sequence

            #self.pembed_wrapper.prepare(inputs_embeds = embeds, positions_embeds = pembeds)
            model_logits = self.model(inputs_embeds = embeds).logits #, position_ids = pembeds).logits
            next_logits = model_logits[...,-1,:]
            next_embeds = logits2embeds(next_logits)
            provision = output(next_logits)
            yield next_logits, provision
            #logits = torch.cat([logits, next_logits[...,None,:]], dim=-2)
            #next_ids = self.logits2ids(next_logits)
            #ids = torch.cat([ids, next_ids[...,None,:]], dim=-2)
            embeds = torch.cat([embeds, next_embeds[...,None,:]], dim=-2)
            #next_pembeds = self.pembed(torch.full(pembeds.shape[:-2],pembeds.shape[-2],device=self.model.device))
            #pembeds = torch.cat([pembeds, next_pembeds[...,None,:]], dim=-2)
        #    if handler is not None and False is handler(logits=logits, ids=ids, embeds=embeds):
        #        break
        #return output(logits)

    def loss(self, input_embeds, label_probs):
        # OUT OF GPU RAM ISSUES COULD BE ADDRESSED VIA GRADIENT CHECKPOINTING
        embeds = self.inputs2embeds(input_embeds)
        model_logits = self.model(inputs_embeds = embeds).logits

        seq_len, vocab_size = model_logits.shape[-2:]
        model_logits = model_logits[...,:-1,:].reshape(-1, vocab_size)

        if label_probs.shape == input_embeds.shape[:-1]:
            label_probs = label_probs[...,1:].reshape(-1)
        else:
            label_probs = label_probs[...,1:,:].reshape(-1, vocab_size)

        if torch.has_cuda:
            if label_probs.device.type == 'cpu':
                label_probs = label_probs.cuda()

        return torch.nn.functional.cross_entropy(model_logits, label_probs)


        # i tried to use distance from embeddings, but ran into cognitive issues.
        # it probably just needs some basic debugging
        # if it's too big, gradient checkpointing could be used
#        model_probs = self.logits2probs(model_logits)
#        
#        weighted_distances = torch.tensor(0, device=embeds.device, dtype=embeds.dtype, requires_grad=True)#torch.zeros(size=embeds.shape[:-1], device=embeds.device)
#
#        for dim in range(self.embed.embedding_dim):
#            # calculate the difference between each label embedding, and each vocab embedding, on this dimension
#            dim_distance = self.embed.weight[(None,)*(len(embeds.shape)-1)+(...,dim)] - embeds[...,dim,None]#.expand(*embeds.shape[:-1],self.embed.num_embeddings)
#            # weighted squared difference
#            dim_distance = dim_distance * dim_distance * model_probs
#            weighted_distances += dim_distance.sum()
#            #id_distance = input_embeds[...,id:id+block_size,:] - vocab_embeds[...,id:id+block_size,:]
#            
#            
#        
##        if self.expanded_embeddings is None:
##            self.expanded_embeddings = self.embed(torch.arange(self.embed.num_embeddings, device=self.model.device))
##
##        comparison_shape = (*embeds.shape[:-1], self.embed.num_embeddings, self.embed.embedding_dim)
##
##        # expand embeds for each vocab word to be weighted
##        input_embeds = embeds[...,None,:].expand(comparison_shape)
##        # expand vocab embeds for each batch and token
##        vocab_embeds = self.expanded_embeddings.expand(comparison_shape)
##
##        # calculate distance and weight the result
##        block_size = 128
##        for id in range(0,self.embed.num_embeddings,block_size):
##            # calculate the difference between each label embedding, and each vocab embedding, a vector product
##            id_distance = input_embeds[...,id:id+block_size,:] - vocab_embeds[...,id:id+block_size,:]
##            # turn this difference into a distance in embedding space
##            id_distance = (id_distance * id_distance)
##            id_distance = id_distance.sum(-1)
##            id_distance = id_distance.sqrt()
##            # weight each distance by the model output, and sum them
##            weighted_distances = weighted_distance + (id_distance * model_probs[...,id:id+block_size]).sum()
##
##        # rather than iterating over the vocab here, would be better to iterate over batches or such
##        # 
##        
        return weighted_distances.log()

    def inputs2embeds(self, inputs):#, pinputs = None):
        #inputs, pinputs = self.inputs2ids(inputs, pinputs)
        inputs = self.inputs2ids(inputs)
        if not inputs.dtype.is_floating_point:
            inputs = self.ids2embeds(inputs)
        assert inputs.shape[-1] == self.embed.embedding_dim
        #if not pinputs.dtype.is_floating_point:
        #    pinputs = self.pembed(pinputs)
        #assert pinputs.shape[-1] == self.pembed.embedding_dim
        return inputs#, pinputs
    def inputs2ids(self, inputs):#, pinputs = None):
        if type(inputs) is str or (type(inputs) is list and (not len(inputs) or type(inputs[0]) is str)):
            inputs = self.strs2ids(inputs)
            if inputs.dtype.is_floating_point: # sometimes returned when data is empty
                inputs = inputs.to(torch.long)
        #if pinputs is None:
        #    if inputs.dtype.is_floating_point:
        #        pinputs = torch.arange(inputs.shape[-2]).expand(inputs.shape[:-1])
        #    else:
        #        pinputs = torch.arange(inputs.shape[-1]).expand(inputs.shape)
        if torch.has_cuda:
            if inputs.device.type == 'cpu':
                inputs = inputs.cuda()
            #if pinputs.device.type == 'cpu':
            #    pinputs = pinputs.cuda()
        return inputs#, pinputs

    def train(self, optimiser=None, name=None, **optim_kwparams):
        prev_training = self.model.training
        prev_grad = torch.is_grad_enabled()
        self.model.train()
        torch.set_grad_enabled(True)
        class Trainer:
            def __enter__(trainer):
                return trainer
            def __exit__(trainer, *params):
                if not prev_training:
                    self.model.eval()
                torch.set_grad_enabled(prev_grad)
                if name is not None:
                    self.model.save_pretrained(name)
            if optimiser is not None:
                optim = optimiser(params=self.model.parameters(), **optim_kwparams)
                def epoch(trainer):
                    class Epoch:
                        def __enter__(epoch):
                            return epoch
                        def __enter__(epoch):
                            Trainer.optim.zero_grad()
                            #model.zero_grad()
                        def __exit__(epoch):
                            Trainer.optim.step()
        return Trainer()

    def eval(self):
        prev_training = self.model.training
        prev_grad = torch.is_grad_enabled()
        self.model.eval()
        torch.set_grad_enabled(False)
        class Evaluator:
            def __enter__(evaluator):
                return evaluator
            def __exit__(evaluator, *params):
                if prev_training:
                    self.model.train()
                torch.set_grad_enabled(prev_grad)
        return Evaluator()

def HF(name):
    class Pretrained(EmbeddingGeneration):
        def __init__(self, model=name, tokenizer=name):
            try:
                super().__init__(model, tokenizer)
            except KeyError as kerr:
                if kerr.args == ('codegen',):
                    kerr.args = ('codegen', 'As of 2022-06-13, codegen is available in git+https://github.com/rooa/transformers.git@add_codegen')
                raise kerr
    Pretrained.__name__ = name
    return Pretrained

GPT2 = HF('gpt2')

CodeParrot = HF('lvwerra/codeparrot')
CodeParrot_small = HF('lvwerra/codeparrot-small')
CodeParrot.small = CodeParrot_small
CodeParrot.large = CodeParrot

InCoder1B = HF('facebook/incoder-1B')
InCoder6B = HF('facebook/incoder-6B')
InCoder = InCoder6B
InCoder.small = InCoder1B
InCoder.large = InCoder6B

CodeGen350m = HF('Salesforce/codegen-350M-mono')
CodeGen2B = HF('Salesforce/codegen-2B-mono')
CodeGen6B = HF('Salesforce/codegen-6B-mono')
CodeGen16B = HF('Salesforce/codegen-16B-mono')
CodeGen = CodeGen16B
CodeGen.small = CodeGen350m
CodeGen.medium = CodeGen2B
CodeGen.large = CodeGen6B
CodeGen.huge = CodeGen16B

if __name__ == '__main__':
    code = CodeParrot.small()
    from sys import stdout
    with code.eval():
        stdout.write('def')
        stdout.flush()
        for logits, word in code.generate('def'):
            if word.startswith('\n\n'):
                stdout.write('\n\n')
                break
            stdout.write(word)
            stdout.flush()
    #print(code.generate('def', 32))

#    model = CodeParrot.small() #gpt2 = GPT2()
#    #logits = gpt2.generate("Once upon", 16, logits2embeds=gpt2.sum, output=gpt2.logits2logits)
#    ## see if we can reduce loss
#    ## initialise optimizer, put models in training mode
#    #print(gpt2.logits2strs(logits))
#    #loss = gpt2.loss(gpt2.sum(logits))
#    #print(loss)
#    ##output = gpt2.generate("Hello world,", 16, logits2embeds=gpt2.sum)
#    ##print(output)
#
#    #import pdb; pdb.set_trace()
#    batch_size=16
#    prompt_len=4
#    gen_len=16
#    with model.train(torch.optim.AdamW, lr=0.0001) as training:
#        while True:
#            #with torch.no_grad():
#                # here we generate data. it's just prompts right now but i meant to just use them as a seed and generate
#                prompts = torch.randint(0, model.embed.num_embeddings, (batch_size,prompt_len))
#                pre_embeds = model.inputs2embeds(prompts)
#                # 
#                post_logits = model.process(pre_embeds, output=model.logits2logits, prefix=False)[...,:-1,:]
#                post_embeds = model.sum(post_logits)
#                    # for the sum idea, we would want these post_embeds to generate a further token distribution
#                    # the same as previously happened without them.
#                    # this is present in the prompt data, but it's changing which is unideal.
#                    ## anyway, so we want the post_embeds to match the pre_embeds, roughly.
#                    #    # where pre_embeds is actually 
#                #logits, sum = next(model.generate(prompts, gen_len + prompt_len, output=model.sum))
#
#                # this uses probabilities of every token
#                loss = model.loss(post_embeds, prompts[...,1:])
#                loss.backward()
#                print(loss)
                
