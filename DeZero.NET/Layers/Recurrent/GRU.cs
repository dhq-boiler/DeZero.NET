using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Layers.Recurrent
{
    public class GRU : Layer
    {
        private Queue<Variable> _stateHistory = new Queue<Variable>();
        private const int MAX_STATE_HISTORY = 60; // シーケンス長に合わせる
        public Property<DeZero.NET.Layers.Linear.Linear> Wxz { get; } = new(nameof(Wxz)); 
        public Property<DeZero.NET.Layers.Linear.Linear> Wxr { get; } = new(nameof(Wxr));  
        public Property<DeZero.NET.Layers.Linear.Linear> Wxh { get; } = new(nameof(Wxh));  
        public Property<DeZero.NET.Layers.Linear.Linear> Whz { get; } = new(nameof(Whz)); 
        public Property<DeZero.NET.Layers.Linear.Linear> Whr { get; } = new(nameof(Whr));
        public Property<DeZero.NET.Layers.Linear.Linear> Whh { get; } = new(nameof(Whh));
        public Property<Variable> H { get; } = new(nameof(H));

        public GRU(int inSize, int hiddenSize)
        {
            RegisterEvent(Wxz, Wxr, Wxh, Whz, Whr, Whh, H);

            Wxz.Value = new DeZero.NET.Layers.Linear.Linear(in_size: inSize, out_size: hiddenSize);
            SetAttribute("Wxz", Wxz.Value);
            Wxr.Value = new DeZero.NET.Layers.Linear.Linear(in_size: inSize, out_size: hiddenSize);
            SetAttribute("Wxr", Wxr.Value);
            Wxh.Value = new DeZero.NET.Layers.Linear.Linear(in_size: inSize, out_size: hiddenSize);
            SetAttribute("Wxh", Wxh.Value);
            Whz.Value = new DeZero.NET.Layers.Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
            SetAttribute("Whz", Whz.Value);
            Whr.Value = new DeZero.NET.Layers.Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
            SetAttribute("Whr", Whr.Value);
            Whh.Value = new DeZero.NET.Layers.Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
            SetAttribute("Whh", Whh.Value);
        }

        public override Variable[] Forward(params Variable[] variables)
        {
            try
            {
                var x = variables[0];
                var batchSize = x.Shape[0];

                // 状態の初期化または再利用
                if (H.Value == null || H.Value.Shape[0] != batchSize)
                {
                    CleanupOldStates(); // 古い状態をクリーンアップ
                    H.Value = xp.zeros(new Shape(batchSize, Wxz.Value.OutSize.Value), dtype: Dtype.float32).ToVariable();
                }

                // 既存の計算処理
                var result = CalculateNextState(x);

                // 状態履歴の管理
                ManageStateHistory(result);

                return [result];
            }
            catch (Exception ex)
            {
                Console.WriteLine($"GRU Forward Error: {ex.Message}");
                throw;
            }
        }

        private Variable CalculateNextState(Variable x)
        {
            using (var scope = new ComputationScope())
            {
                var wxz = scope.Register(Wxz.Value.Forward(x)[0]);
                var whz = scope.Register(Whz.Value.Forward(H.Value)[0]);
                var z = scope.Register(DeZero.NET.Functions.Sigmoid.Invoke(
                    DeZero.NET.Functions.Add.Invoke(wxz, whz).Item1[0])[0]);

                var wxr = scope.Register(Wxr.Value.Forward(x)[0]);
                var whr = scope.Register(Whr.Value.Forward(H.Value)[0]);
                var r = scope.Register(DeZero.NET.Functions.Sigmoid.Invoke(
                    DeZero.NET.Functions.Add.Invoke(wxr, whr).Item1[0])[0]);

                var wxh = scope.Register(Wxh.Value.Forward(x)[0]);
                var rh = scope.Register(DeZero.NET.Functions.Mul.Invoke(r, H.Value)[0]);
                var whh = scope.Register(Whh.Value.Forward(rh)[0]);
                var h_tilde = scope.Register(DeZero.NET.Functions.Tanh.Invoke(
                    DeZero.NET.Functions.Add.Invoke(wxh, whh).Item1[0])[0]);

                return DeZero.NET.Functions.Add.Invoke(
                    scope.Register(DeZero.NET.Functions.Mul.Invoke(z, H.Value)[0]),
                    scope.Register(DeZero.NET.Functions.Mul.Invoke(
                        DeZero.NET.Functions.Sub.Invoke(xp.array(1).ToVariable(), z)[0],
                        h_tilde)[0])
                ).Item1[0];
            }
        }
        private void ManageStateHistory(Variable newState)
        {
            _stateHistory.Enqueue(newState);
            if (_stateHistory.Count > MAX_STATE_HISTORY)
            {
                var oldState = _stateHistory.Dequeue();
                oldState?.Dispose();
            }
        }

        private void CleanupOldStates()
        {
            while (_stateHistory.Count > 0)
            {
                var state = _stateHistory.Dequeue();
                state?.Dispose();
            }
            H.Value?.Dispose();
            H.Value = null;
        }

        public void ResetState()
        {
            CleanupOldStates();
        }
    }
}
